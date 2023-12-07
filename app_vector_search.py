import os
import sys

import gradio as gr
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from mylangchain.embeddings import CohereEmbeddings
from mylangchain.vectorstores.oracleaivector import OracleAIVector

sys.path.append('../..')

# read local .env file
_ = load_dotenv(find_dotenv())

embedding_search_document = CohereEmbeddings(model="embed-multilingual-v3.0", input_type="search_document")
embedding_search_query = CohereEmbeddings(model="embed-multilingual-v3.0", input_type="search_query")

ORACLE_AI_VECTOR_CONNECTION_STRING = os.environ["ORACLE_AI_VECTOR_CONNECTION_STRING"]


def load_document(file_text):
    """
    Specify a DocumentLoader to load in your unstructured data as Documents.
    A Document is a dict with text (page_content) and metadata.
    """
    loader = TextLoader(file_text.name)
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


def embed_document(copy_of_first_trunk_content_text, copy_of_last_trunk_content_text):
    """
    To be able to look up our document splits, we first need to store them where we can later look them up.
    The most common way to do this is to embed the contents of each document split.
    We store the embedding and splits in a vectorstore.
    """
    first_trunk_vector_text = embedding_search_document.embed_documents([copy_of_first_trunk_content_text])
    last_trunk_vector_text = embedding_search_document.embed_documents([copy_of_last_trunk_content_text])

    OracleAIVector.from_documents(
        embedding=embedding_search_document,
        documents=all_splits,
        collection_name="docs_vector_search",
        connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
        pre_delete_collection=True,  # Overriding a vectorstore
    )

    return gr.Textbox(value=first_trunk_vector_text), gr.Textbox(value=last_trunk_vector_text)


# @tool
def chat_document_stream(question2_text):
    """
    Retrieve relevant splits for any question using similarity search.
    This is simply "top K" retrieval where we select documents based on embedding similarity to the query.
    """
    # Use OracleAIVector
    vectorstore = OracleAIVector(connection_string=ORACLE_AI_VECTOR_CONNECTION_STRING,
                                 collection_name="docs_vector_search",
                                 embedding_function=embedding_search_query,
                                 )
    docs_dataframe = []

    all_docs = vectorstore.similarity_search_with_score("q", k=20)
    for doc, _ in all_docs:
        # print(f"{doc}")
        splits = doc.page_content.split(" ")
        # print(f"{splits}")
        docs_dataframe.append([splits[0], splits[1], splits[2]])

    docs = vectorstore.similarity_search_with_score(question2_text, k=1)
    # print(f"docs: {docs}")
    for doc, _ in docs:
        message = doc.page_content.split(" ")[0]
    return gr.Dataframe(value=docs_dataframe, wrap=True), gr.Textbox(message)


with gr.Blocks() as app:
    gr.Markdown(value="# Oracle AI Vector Search デモ")

    with gr.TabItem(label="Step-1.ドキュメントの読み込み"):
        with gr.Row():
            with gr.Column():
                page_count_text = gr.Textbox(label="ページ数", lines=1)
        with gr.Row():
            with gr.Column():
                page_content_text = gr.Textbox(label="ページ内容", lines=15, max_lines=15, autoscroll=False,
                                               show_copy_button=True)
        with gr.Row():
            with gr.Column():
                file_text = gr.File(label="ファイル", file_types=[".txt"], type="file")
        with gr.Row():
            with gr.Column():
                gr.Examples(
                    examples=[os.path.join(os.path.dirname(__file__), "files/text_search_vs_vector_search.txt")],
                    label="ファイル事例", inputs=file_text)

        with gr.Row():
            with gr.Column():
                load_button = gr.Button(value="読み込み", label="load", variant="primary")

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
                chunk_size_text = gr.Textbox(label="チャンク・サイズ(Chunk Size)", lines=1, value="20")
            with gr.Column():
                chunk_overlap_text = gr.Textbox(label="チャンク・オーバーラップ(Chunk Overlap)", lines=1,
                                                value="0")
        with gr.Row():
            with gr.Column():
                gr.Examples(examples=[[20, 0], [50, 0], [100, 0], [200, 0]],
                            inputs=[chunk_size_text, chunk_overlap_text])
        with gr.Row():
            with gr.Column():
                split_button = gr.Button(value="分割", label="Split", variant="primary")

    with gr.TabItem(label="Step-3.ベクトル・データベースへ保存"):
        with gr.Row():
            with gr.Column():
                copy_of_first_trunk_content_text = gr.Textbox(label="最初の Chunk 内容", lines=10, max_lines=10,
                                                              autoscroll=False, interactive=False)
                first_trunk_vector_text = gr.Textbox(label="ベクトル化後の Chunk 内容", lines=10, max_lines=10,
                                                     autoscroll=False, show_copy_button=True)
        with gr.Row():
            with gr.Column():
                copy_of_last_trunk_content_text = gr.Textbox(label="最後の Chunk 内容", lines=10, max_lines=10,
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
                    headers=["ページ・コンテンツ", "テキスト・検索", "ベクトル・検索"],
                    datatype=["str", "str", "str"],
                    row_count=2,
                    col_count=(3, "fixed"),
                    wrap=True
                )
        with gr.Row():
            with gr.Column():
                answer2_text = gr.Textbox(label="回答", lines=2, max_lines=2,
                                          autoscroll=False, interactive=False, show_copy_button=True)
        with gr.Row():
            with gr.Column():
                question2_text = gr.Textbox(label="質問", lines=1)
        with gr.Row():
            with gr.Column():
                gr.Examples(examples=[
                    "自動運転",
                    "ネコカフェ",
                    "AI",
                    "下げた",
                    "桜",
                    "運動",
                    "赤字",
                    "フットボール"],
                    inputs=question2_text)
        with gr.Row():
            with gr.Column():
                chat_document_button = gr.Button(value="送信", label="chat_document", variant="primary")

    load_button.click(load_document,
                      inputs=[file_text],
                      outputs=[page_count_text, page_content_text],
                      )

    split_button.click(split_document,
                       inputs=[chunk_size_text, chunk_overlap_text],
                       outputs=[chunk_count_text, first_trunk_content_text, last_trunk_content_text,
                                copy_of_first_trunk_content_text, copy_of_last_trunk_content_text],
                       )

    embed_and_save_button.click(embed_document,
                                inputs=[copy_of_first_trunk_content_text, copy_of_last_trunk_content_text],
                                outputs=[first_trunk_vector_text, last_trunk_vector_text],
                                )

    chat_document_button.click(chat_document_stream,
                               inputs=[question2_text],
                               outputs=[answer2_dataframe, answer2_text])

app.queue()
if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7863)
