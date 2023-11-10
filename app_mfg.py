import os
from typing import List

import openai
import sys
from dotenv import load_dotenv, find_dotenv

# from curl_cffi import requests
from langchain.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings import CohereEmbeddings
from mylangchain.embeddings import CohereEmbeddings
from langchain.schema import AIMessage, HumanMessage, SystemMessage, Document
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.tools import tool

import gradio as gr
from langchain.vectorstores.pgvector import PGVector

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

sys.path.append('../..')

# read local .env file
_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']
openai.api_base = os.environ['OPENAI_API_BASE']
os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"

persist_directory = './docs/chroma/'
"""
"search_document": Use this when you encode documents for embeddings that you store in a vector database for search use-cases.
"search_query": Use this when you query your vector DB to find relevant documents.
"classification": Use this when you use the embeddings as an input to a text classifier.
"clustering": Use this when you want to cluster the embeddings.
"""
embedding_search_document = CohereEmbeddings(model="embed-multilingual-v3.0", input_type="search_document")
embedding_search_query = CohereEmbeddings(model="embed-multilingual-v3.0", input_type="search_query")
# embedding_search_document = OpenAIEmbeddings()
# embedding_search_query = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# PGVector needs the connection string to the database.
CONNECTION_STRING = os.environ["CONNECTION_STRING"]

ORACLE_DB_CONNECT_STRING = os.environ['ORACLE_DB_CONNECT_STRING']

# print(sqlalchemy.__version__)
# 创建引擎
engine = create_engine(ORACLE_DB_CONNECT_STRING, echo=False)


def chat_stream(question1_text):
    messages = [
        SystemMessage(
            content="You are a helpful assistant."
        ),
        HumanMessage(
            content=question1_text
        ),
    ]
    result = llm(messages)

    return gr.Textbox(result.content)


def save_report(employee_name_text, task_report_text):
    insert_stmt = text(
        "INSERT INTO daily_task_report (employee_name, task_report) VALUES (:employee_name, :task_report)")

    with Session(engine) as session:
        try:
            result = session.execute(insert_stmt,
                                     {"employee_name": employee_name_text, "task_report": task_report_text})
            print(f"========= {result.rowcount} rows inserted. =========")
            session.commit()
        except Exception as e:
            print(f"=========\n {e} \n=========")
            session.rollback()

    return gr.Textbox(value="保存しました", visible=True)


def load_document_from_db():
    """
    Specify a DocumentLoader to load in your unstructured data as Documents.
    A Document is a dict with text (page_content) and metadata.
    """
    documents: List[Document] = []
    select_stmt = text("SELECT employee_name, task_report FROM daily_task_report")
    with Session(engine) as session:
        try:
            result = session.execute(select_stmt)
            for row in result:
                document = Document(page_content=f"{row.employee_name}: {row.task_report}",
                                    metadata={"source": "Oracle データベース"})
                documents.append(document)
            session.commit()
        except Exception as e:
            print(f"=========\n {e} \n=========")
            session.rollback()

    global data
    data = documents
    print(f"data: {data}")
    page_count = len(data)
    for doc in data:
        page_content_text = doc.page_content
        while "\n\n" in page_content_text:
            page_content_text = page_content_text.replace("\n\n", "\n")
        doc.page_content = page_content_text

    doc_str = '\n'.join(str(doc.page_content) for doc in documents)

    return gr.Textbox(value=str(page_count)), gr.Textbox(value=doc_str)


def load_document(file_text, web_page_url_text):
    """
    Specify a DocumentLoader to load in your unstructured data as Documents.
    A Document is a dict with text (page_content) and metadata.
    """
    if web_page_url_text == "" or web_page_url_text is None:
        loader = TextLoader(file_text.name)
    else:
        loader = WebBaseLoader(web_page_url_text)

    global data
    data = loader.load()
    # print(f"data: {data}")
    page_count = len(data)
    page_content_text = data[0].page_content
    while "\n\n" in page_content_text:
        page_content_text = page_content_text.replace("\n\n", "\n")
    data[0].page_content = page_content_text

    return gr.Textbox(value=str(page_count)), gr.Textbox(value=page_content_text)


def split_document(chunk_size_text, chunk_overlap_text):
    """
    Split the Document into chunks for embedding and vector storage.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=int(chunk_size_text),
                                                   chunk_overlap=int(chunk_overlap_text))

    global data, all_splits
    all_splits = text_splitter.split_documents(data)
    # print(f"all_splits: {all_splits}")
    chunk_count_text = len(all_splits)
    first_trunk_content_text = all_splits[0].page_content
    last_trunk_content_text = all_splits[-1].page_content

    return gr.Textbox(value=str(chunk_count_text)), gr.Textbox(value=first_trunk_content_text), gr.Textbox(
        value=last_trunk_content_text), gr.Textbox(value=first_trunk_content_text), gr.Textbox(
        value=last_trunk_content_text)


def embed_document(cope_of_first_trunk_content_text, cope_of_last_trunk_content_text):
    """
    To be able to look up our document splits, we first need to store them where we can later look them up.
    The most common way to do this is to embed the contents of each document split.
    We store the embedding and splits in a vectorstore.
    """
    first_trunk_vector_text = embedding_search_document.embed_documents([cope_of_first_trunk_content_text])
    last_trunk_vector_text = embedding_search_document.embed_documents([cope_of_last_trunk_content_text])
    if os.path.exists(persist_directory):
        if len(os.listdir(persist_directory)) > 0:
            for root, dirs, files in os.walk(persist_directory, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(persist_directory)
        else:
            os.rmdir(persist_directory)
    # Use Chroma
    # Chroma.from_documents(persist_directory=persist_directory,
    #                       collection_name="docs",
    #                       documents=all_splits,
    #                       embedding=embedding_search_document)
    # Use PGVector
    PGVector.from_documents(
        embedding=embedding_search_document,
        documents=all_splits,
        collection_name="docs",
        connection_string=CONNECTION_STRING,
        pre_delete_collection=True,  # Overriding a vectorstore
    )
    # print(f"vectorstore: {vectorstore}")

    return gr.Textbox(value=first_trunk_vector_text), gr.Textbox(value=last_trunk_vector_text)


@tool
def chat_document_stream(question2_text):
    """
    Retrieve relevant splits for any question using similarity search.
    This is simply "top K" retrieval where we select documents based on embedding similarity to the query.
    """
    # Use Chroma
    # vectorstore = Chroma(persist_directory=persist_directory, collection_name="docs",
    #                      embedding_function=embedding_search_query)
    # Use PGVector
    vectorstore = PGVector(connection_string=CONNECTION_STRING,
                           collection_name="docs",
                           embedding_function=embedding_search_query,
                           )
    docs_dataframe = []
    docs = vectorstore.similarity_search_with_score(question2_text)
    # print(f"len(docs): {len(docs)}")
    for doc, score in docs:
        # print(f"doc: {doc}")
        # print("Score: ", score)
        docs_dataframe.append([doc.page_content, doc.metadata["source"]])

    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use ten sentences maximum and keep the answer as concise as possible.
    Don't try to answer anything that isn't in context.  
    {context}
    Question: {question}
    Helpful Answer:"""
    rag_prompt_custom = PromptTemplate.from_template(template)
    # Method-1
    # retriever = vectorstore.as_retriever()
    #
    # rag_chain = (
    #         {"context": retriever, "question": RunnablePassthrough()} | rag_prompt_custom | llm
    # )
    #
    # result = rag_chain.invoke(question2_text)
    # # print(result)
    #
    # return gr.Dataframe(value=docs_dataframe), gr.Textbox(result.content)

    # Method-2
    message = rag_prompt_custom.format_prompt(context=docs, question=question2_text)
    result = llm(message.to_messages())
    return gr.Dataframe(value=docs_dataframe), gr.Textbox(result.content)


with gr.Blocks() as app:
    gr.Markdown(value="# RAG デモ")

    with gr.Tabs() as tabs:
        with gr.TabItem(label="Step--1.チャット"):
            with gr.Row():
                with gr.Column():
                    answer1_text = gr.Textbox(label="回答", lines=15, max_lines=15,
                                              autoscroll=False, interactive=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    question1_text = gr.Textbox(label="質問", lines=1)
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=["東京太郎は何台の車を製造しましたか？",
                                          "大阪太郎は何台の車を製造しましたか？",
                                          "鈴木保奈美さんの誕生日を教えてください",
                                          "鈴木保奈美さんの出身を教えてください",
                                          "鈴木保奈美さんの主な作品を教えてください",
                                          ],
                                inputs=question1_text)
            with gr.Row():
                with gr.Column():
                    chat_button = gr.Button(value="送信", label="chat", variant="primary")

        with gr.TabItem(label="Step-0.作業日報"):
            with gr.Row():
                with gr.Column():
                    save_message_text = gr.Textbox(show_label=False, visible=False)
            with gr.Row():
                with gr.Column():
                    employee_name_text = gr.Textbox(label="名前", lines=1)
            with gr.Row():
                with gr.Column():
                    task_report_text = gr.Textbox(label="作業", lines=15, max_lines=15,
                                                  autoscroll=False, interactive=True, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=[["東京太郎", "2023年11月10日は5台車を製造しました。"],
                                          ["大阪太郎", "2023年11月10日は6台車を製造しました。"],
                                          ["京都太郎", "2023年11月10日は7台車を製造しました。"],
                                          ["北海道太郎", "2023年11月10日は8台車を製造しました。"],
                                          ["奈良太郎", "2023年11月10日は9台車を製造しました。"],],
                                inputs=[employee_name_text, task_report_text])
            with gr.Row():
                with gr.Column():
                    save_report_button = gr.Button(value="保存", label="save", variant="primary")

        with gr.TabItem(label="Step-1.ドキュメントのロード"):
            with gr.Row():
                with gr.Column():
                    page_count_text = gr.Textbox(label="ページ数", lines=1)
            with gr.Row():
                with gr.Column():
                    page_content_text = gr.Textbox(label="ページ内容", lines=15, max_lines=15, autoscroll=False,
                                                   show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    load_from_db_button = gr.Button(value="DBからロード", label="load_from_db", variant="primary")

            with gr.Row(visible=False):
                with gr.Column():
                    file_text = gr.File(label="ファイル", file_types=[".txt"], type="file")
                with gr.Column():
                    web_page_url_text = gr.Textbox(label="ウェブ・ページ", lines=1)
            with gr.Row(visible=False):
                with gr.Column():
                    gr.Examples(examples=[os.path.join(os.path.dirname(__file__), "files/suzukihonami.txt")],
                                label="ファイル事例",
                                inputs=file_text)
                with gr.Column():
                    gr.Examples(examples=["https://ja.wikipedia.org/wiki/東京ラブストーリー",
                                          "https://ja.wikipedia.org/wiki/ニュースの女",
                                          "https://ja.wikipedia.org/wiki/鈴木保奈美",
                                          "https://ja.wikipedia.org/wiki/木村拓哉"],
                                label="ウェブ・ページ事例",
                                inputs=web_page_url_text)

            with gr.Row(visible=False):
                with gr.Column():
                    load_button = gr.Button(value="ロード", label="load", variant="primary")

        with gr.TabItem(label="Step-2.ドキュメントの分割"):
            with gr.Row():
                with gr.Column():
                    chunk_count_text = gr.Textbox(label="Chunk 数", lines=1)
            with gr.Row():
                with gr.Column():
                    first_trunk_content_text = gr.Textbox(label="最初の Chunk 内容", lines=10, max_lines=10,
                                                          autoscroll=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    last_trunk_content_text = gr.Textbox(label="最後の Chunk 内容", lines=10, max_lines=10,
                                                         autoscroll=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    chunk_size_text = gr.Textbox(label="チャンク・サイズ(Chunk Size)", lines=1, value="500")
                with gr.Column():
                    chunk_overlap_text = gr.Textbox(label="チャンク・オーバーラップ(Chunk Overlap)", lines=1,
                                                    value="100")
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=[[50, 0], [200, 0], [500, 0], [500, 100]],
                                inputs=[chunk_size_text, chunk_overlap_text])
            with gr.Row():
                with gr.Column():
                    split_button = gr.Button(value="分割", label="Split", variant="primary")

        with gr.TabItem(label="Step-3.ベクトル・データベースへ保存"):
            with gr.Row():
                with gr.Column():
                    cope_of_first_trunk_content_text = gr.Textbox(label="最初の Chunk 内容", lines=10, max_lines=10,
                                                                  autoscroll=False, interactive=False)
                    first_trunk_vector_text = gr.Textbox(label="ベクトル化後の Chunk 内容", lines=10, max_lines=10,
                                                         autoscroll=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    cope_of_last_trunk_content_text = gr.Textbox(label="最後の Chunk 内容", lines=10, max_lines=10,
                                                                 autoscroll=False, interactive=False)
                    last_trunk_vector_text = gr.Textbox(label="ベクトル化後の Chunk 内容", lines=10, max_lines=10,
                                                        autoscroll=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    embed_and_save_button = gr.Button(value="ベクトル化して保存", label="embed_and_save",
                                                      variant="primary")

        with gr.TabItem(label="Step-4.ドキュメントとチャット"):
            with gr.Row():
                with gr.Column():
                    answer2_dataframe = gr.Dataframe(
                        headers=["ページ・コンテンツ", "ソース"],
                        datatype=["str", "str"],
                        row_count=5,
                        col_count=(2, "fixed"),
                    )
            with gr.Row():
                with gr.Column():
                    answer2_text = gr.Textbox(label="回答", lines=15, max_lines=15,
                                              autoscroll=False, interactive=False, show_copy_button=True)
            with gr.Row():
                with gr.Column():
                    question2_text = gr.Textbox(label="質問", lines=1)
            with gr.Row():
                with gr.Column():
                    gr.Examples(examples=["東京太郎は何台の車を製造しましたか？",
                                          "大阪太郎は何台の車を製造しましたか？",
                                          "鈴木保奈美さんの誕生日を教えてください",
                                          "鈴木保奈美さんの出身を教えてください",
                                          "鈴木保奈美さんの主な作品を教えてください"],
                                inputs=question2_text)
            with gr.Row():
                with gr.Column():
                    chat_document_button = gr.Button(value="送信", label="chat_document", variant="primary")

        chat_button.click(chat_stream,
                          inputs=[question1_text],
                          outputs=[answer1_text])

        save_report_button.click(save_report,
                                 inputs=[employee_name_text, task_report_text],
                                 outputs=[save_message_text])

        load_from_db_button.click(load_document_from_db,
                                  inputs=[],
                                  outputs=[page_count_text, page_content_text],
                                  )

        load_button.click(load_document,
                          inputs=[file_text, web_page_url_text],
                          outputs=[page_count_text, page_content_text],
                          )

        split_button.click(split_document,
                           inputs=[chunk_size_text, chunk_overlap_text],
                           outputs=[chunk_count_text, first_trunk_content_text, last_trunk_content_text,
                                    cope_of_first_trunk_content_text, cope_of_last_trunk_content_text],
                           )

        embed_and_save_button.click(embed_document,
                                    inputs=[cope_of_first_trunk_content_text, cope_of_last_trunk_content_text],
                                    outputs=[first_trunk_vector_text, last_trunk_vector_text],
                                    )

        chat_document_button.click(chat_document_stream,
                                   inputs=[question2_text],
                                   outputs=[answer2_dataframe, answer2_text])

app.queue()
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=17860)
