# 双塔从训练到 Go 调用的最小示例

本流程使用项目内 Python 双塔训练与服务，**Go 端统一使用 NewTorchServeClient** 调用（与 TorchServe 协议一致，降低心智负担）。

## 1. 字段一致性说明

**train_two_tower.py** 与 **two_tower_model_loader.py** 字段保持一致：

| 来源 | 字段 | 说明 |
|------|------|------|
| `two_tower_meta.json` | `user_cols`, `item_cols`, `embed_dim` | 训练脚本写入，加载器读取；名称与顺序一致 |
| checkpoint `config` | `user_dim`, `item_dim`, `user_layers`, `item_layers` | 训练时保存，加载器用于构建 TwoTower 结构 |

训练脚本默认：`USER_COLS = ["user_age", "user_gender"]`，`ITEM_COLS = ["item_ctr", "item_cvr", "item_price"]`，`EMBED_DIM = 64`，塔结构 `[128, 64]`。修改训练脚本时需同步修改 loader 的默认值（或保证 meta/checkpoint 由同一训练产出）。

---

## 2. 训练双塔模型（Python）

在项目根目录执行：

```bash
# 无数据时会自动生成 data/two_tower_data.csv
python train/train_two_tower.py

# 或指定数据与轮数
python train/train_two_tower.py --data data/my_train.csv --epochs 20
```

产出：

- `model/two_tower_model.pt`：模型权重与 config
- `model/two_tower_meta.json`：`user_cols`、`item_cols`、`embed_dim`

---

## 3. 启动双塔推理服务（Python）

```bash
uvicorn service.two_tower_server:app --host 0.0.0.0 --port 8085 --timeout-keep-alive 30
```

服务遵循 **TorchServe 协议**（与 Go `NewTorchServeClient` 一致）：

- **GET /ping**：健康检查（TorchServeClient.Health 使用）
- **POST /predictions/{model_name}**：请求体 `{"data": [{"user_age": 0.5, "user_gender": 1, ...}]}`，响应 `{"predictions": [0.1, ..., 0.64]}`（User Embedding）
- `GET /health`、`POST /user_embedding`、`POST /item_embedding`：辅助与调试

---

## 4. Go 端统一使用 NewTorchServeClient

Python 双塔服务与 TorchServe 使用同一协议，Go 端只认一个客户端：

```go
import (
	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/service"
)

// 用户塔：Python 双塔服务或 TorchServe，统一用 TorchServeClient
userTowerService := service.NewTorchServeClient(
	"http://localhost:8085", // 端点（Python 服务或 TorchServe）
	"two_tower",             // 模型名（Python 路由 /predictions/two_tower）
	service.WithTorchServeTimeout(5*time.Second),
)

twoTowerRecall := recall.NewTwoTowerRecall(
	featureService,
	userTowerService,
	vectorService,
	recall.WithTwoTowerTopK(100),
	recall.WithTwoTowerCollection("item_embeddings"),
	recall.WithTwoTowerMetric("inner_product"),
)
```

确保：

- 特征服务产出的用户特征键名与 `two_tower_meta.json` 中的 `user_cols` 一致（如 `user_age`, `user_gender`）。
- 向量库中已存在 Item Embedding 集合（名称与 `WithTwoTowerCollection` 一致）；Item Embedding 可由 Python 离线脚本调用 `POST /item_embedding` 或 `two_tower_model_loader.get_item_embedding` 批量计算后写入。

---

## 5. 最小可运行顺序小结

1. **训练**：`python train/train_two_tower.py`
2. **启动服务**：`uvicorn service.two_tower_server:app --host 0.0.0.0 --port 8085`
3. **（可选）** 准备向量库：用 Python 调用 `/item_embedding` 或 loader 批量计算 Item Embedding 并写入 Milvus 等，集合名与 Go 中 `WithTwoTowerCollection` 一致。
4. **运行 Go 示例**：`go run examples/two_tower_recall/main.go`（示例中已使用 `NewTorchServeClient("http://localhost:8085", "two_tower")`）。

无论对接 Python 双塔服务还是 TorchServe，Go 端均使用 **NewTorchServeClient**，协议统一。
