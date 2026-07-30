FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 预下载模型（构建时）
RUN python -c "from unsloth import FastLanguageModel; FastLanguageModel.from_pretrained('unsloth/Qwen2.5-7B-Instruct-bnb-4bit', max_seq_length=2048, load_in_4bit=True)"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')"

EXPOSE 7860 8000

CMD ["python", "src/app.py"]
