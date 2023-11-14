# RAGDemo
RAG Demo

## 準備

```
conda create -n ragdemo python=3.10 -y
conda activate ragdemo
```

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -y -c "nvidia/label/cuda-12.1.0" cuda-runtime

python -c "import torch;print(torch.cuda.is_available());"
---
True
```

```
pip install -r requirements.txt
```

## RAG デモの起動

```
python app.py
```

ブラウザーで [http://localhost:7860](http://localhost:7860) を開いて、アクセスしてください。

refer: [https://python.langchain.com/docs/use_cases/question_answering/](https://python.langchain.com/docs/use_cases/question_answering/)


## （Optional） PGVector

PGVector の起動、

```
mkdir -p /root/pg/data; chmod 777 /root/pg/data
docker run --name pgvector --restart=always -p 5454:5432 -v /root/pg/data:/var/lib/postgresql/data -e POSTGRES_INITDB_ARGS="--locale-provider=icu --icu-locale=ja-x-icu" -e POSTGRES_USER=username -e POSTGRES_PASSWORD=password -e POSTGRES_DB=postgres -d ankane/pgvector:v0.5.1
```

拡張機能を有効にする、

```
docker exec -it pgvector bash
psql -U username -d postgres
CREATE EXTENSION vector;
```

データベースの作成、

```
docker exec -it pgvector bash
psql -U username -d postgres
CREATE DATABASE ragdemo;
```

[pgvector document](https://github.com/pgvector/pgvector)

## (Optional) Oracle Database

refer: [https://docs.sqlalchemy.org/en/20/index.html](https://docs.sqlalchemy.org/en/20/index.html)

```
# grant dba to pdbadmin;
```

```
docker exec -it oracledb23c bash
export NLS_LANG=Japanese_Japan.AL32UTF8;
sqlplus pdbadmin/<your_password>@FREEPDB1
```

```
select * from USER_TABLES;

select * from USER_TAB_COMMENTS;

select * from USER_COL_COMMENTS;
```

```
select T1.TABLE_NAME, T2.COMMENTS as TAB_COMMNETS, T3.COLUMN_NAME as COLUMN_NAME, T3.COMMENTS as COL_COMMNETS
from USER_TABLES T1, USER_TAB_COMMENTS T2, USER_COL_COMMENTS T3
where T1.TABLE_NAME = T2.TABLE_NAME
and T1.TABLE_NAME = T3.TABLE_NAME;
```