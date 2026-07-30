.PHONY: setup train evaluate serve test docker clean

setup:
	pip install -r requirements.txt
	python -c "from unsloth import FastLanguageModel; FastLanguageModel.from_pretrained('unsloth/Qwen2.5-7B-Instruct-bnb-4bit', max_seq_length=2048, load_in_4bit=True)"
	python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', device='cpu')"

train:
	python src/train_lora.py

evaluate:
	python src/evaluate_model.py

serve:
	@echo "Starting vLLM server..."
	python -m vllm.entrypoints.openai.api_server \
		--model unsloth/Qwen2.5-7B-Instruct \
		--enable-lora \
		--lora-modules itops=./outputs/lora_final/final_lora \
		--max-lora-rank 16 \
		--gpu-memory-utilization 0.85 \
		--max-model-len 2048 \
		--port 8000 & \
	sleep 10 && python src/app.py

test:
	pytest tests/ -v --tb=short

docker:
	docker build -t it-ops-assistant .
	docker compose up -d

clean:
	rm -rf outputs/ chroma_db/ __pycache__/ .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
