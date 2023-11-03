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