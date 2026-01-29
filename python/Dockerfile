# Python 统一模型推理服务（uv + unified_server）
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

# 系统依赖（部分 Python 包需编译）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只复制依赖声明，便于利用 Docker 层缓存
COPY pyproject.toml .
# 有 uv.lock 时取消下一行注释，并在 RUN 中加上 --locked
# COPY uv.lock .

# 安装依赖（不安装开发依赖）；无 lock 时在构建时解析
ENV UV_NO_DEV=1
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# 复制应用代码
COPY . .

RUN mkdir -p model data

EXPOSE 8080
ENV PORT=8080
ENV HOST=0.0.0.0
ENV PYTHONUNBUFFERED=1
# 统一服务：默认启用所有模型（可在 docker-compose 或运行时覆盖）
ENV ENABLED_MODELS=xgb,deepfm,mmoe,user_tower,item_tower,youtube_dnn,dssm,graph_recall

# 使用 venv 中的 uvicorn
ENV PATH="/app/.venv/bin:$PATH"
# 健康检查：统一服务使用 /ping
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/ping')" || exit 1

CMD ["uvicorn", "service.unified_server:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "30"]
