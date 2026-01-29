# Python 项目结构

```
python/
├── service/           # 推理服务
│   ├── domain/       # 协议约定（TorchServe 请求/响应，与 reckit 一致）
│   ├── app/          # 用例（批量预测等）
│   ├── server.py, deepfm_server.py, mmoe_server.py, two_tower_server.py, ...
│   ├── *_model_loader.py
│   ├── metrics.py, middleware.py
│   └── README.md
├── train/            # 训练脚本与数据/特征
├── tests/, scripts/, data/
└── requirements.txt, Dockerfile, ...
```

协议约束见 Reckit 仓库 [docs/MODEL_SERVICE_PROTOCOL.md](../docs/MODEL_SERVICE_PROTOCOL.md)。
