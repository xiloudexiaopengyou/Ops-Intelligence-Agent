# 部署手册

## 环境要求

| 项目 | 最低要求 |
|------|----------|
| GPU | NVIDIA RTX 4060 8GB |
| CUDA | 12.1+ |
| Python | 3.10+ |
| 磁盘 | 20 GB |
| 内存 | 16 GB RAM |

## Docker 部署

```bash
docker build -t it-ops-assistant .
docker compose up -d
```

## 手动部署

```bash
make setup   # 安装依赖 + 下载模型
make train   # 训练模型
make serve   # 启动 vLLM + Gradio
```

服务地址: Gradio http://localhost:7860 · vLLM API http://localhost:8000
