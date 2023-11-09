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
docker run --name pgvector --restart=always -p 5432:5432 -v /root/pg/data:/var/lib/postgresql/data -e POSTGRES_USER=username -e POSTGRES_PASSWORD=password -e POSTGRES_DB=postgres -d ankane/pgvector:v0.5.1
```

データベースの作成、

```
docker exec -it pgvector bash
psql -U username -d postgres
CREATE DATABASE ragdemo;
```

[pgvector document](https://github.com/pgvector/pgvector)
