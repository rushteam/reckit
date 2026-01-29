package main

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/filter"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/rerank"
	"github.com/rushteam/reckit/store"
)

// demoItemFeatures 为物品注入基础特征（与 Python 训练时的特征对齐）
type demoItemFeatures struct{}

func (n *demoItemFeatures) Name() string        { return "demo.item_features" }
func (n *demoItemFeatures) Kind() pipeline.Kind { return pipeline.KindPostProcess }

func (n *demoItemFeatures) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	// 为物品注入基础特征（与 Python 训练时的特征对齐）
	for i, it := range items {
		it.Features["item_ctr"] = float64(i+1) * 0.1
		it.Features["item_cvr"] = float64(6-i) * 0.05
		it.Features["item_price"] = float64(100 - i*10)
		if i%2 == 0 {
			it.Meta["category"] = "A"
		} else {
			it.Meta["category"] = "B"
		}
	}
	return items, nil
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// 初始化 Store
	memStore := store.NewMemoryStore()
	defer memStore.Close(ctx)

	// 创建特征注入节点
	enrichNode := &feature.EnrichNode{
		UserFeaturePrefix:  "user_",
		ItemFeaturePrefix:  "item_",
		CrossFeaturePrefix: "cross_",
	}

	// 创建 DeepFM RPC 模型（调用 Python 服务）
	// 注意：需要先启动 DeepFM 服务 (python service/deepfm_server.py)
	deepfmModel := model.NewRPCModel("deepfm", "http://localhost:8080/predictions/deepfm", 5*time.Second)

	// 构建 Pipeline（使用 Python DeepFM 模型）
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			// 召回
			&recall.Fanout{
				Sources: []recall.Source{
					&recall.Hot{IDs: []string{"1", "2", "3", "4", "5"}},
				},
				Dedup: true,
			},
			// 过滤
			&filter.FilterNode{
				Filters: []filter.Filter{
					filter.NewUserBlockFilter(nil, "user:block"),
					filter.NewExposedFilter(nil, "user:exposed", 7*24*3600, 0),
				},
			},
			// 注入物品基础特征
			&demoItemFeatures{},
			// 特征注入（将用户特征、物品特征、交叉特征组合）
			enrichNode,
			// 排序（使用 Python DeepFM 模型）
			&rank.RPCNode{Model: deepfmModel},
			// 重排
			&rerank.Diversity{LabelKey: "category"},
		},
	}

	// 创建用户上下文
	rctx := &core.RecommendContext{
		UserID: "42",
		Scene:  "feed",
		UserProfile: map[string]any{
			"age":    25.0,
			"gender": 1.0, // 1=male, 0=female
			"city":   "beijing",
		},
		Realtime: map[string]any{
			"hour":   float64(time.Now().Hour()),
			"device": "mobile",
		},
		Params: map[string]any{
			"debug": true,
		},
	}

	// 运行 Pipeline
	fmt.Println("开始运行 Pipeline（使用 Python DeepFM 模型）...")
	fmt.Println("提示：确保 DeepFM 服务已启动: cd python && uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080")
	fmt.Println()

	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		fmt.Printf("Pipeline 运行失败: %v\n", err)
		fmt.Println("\n请确保:")
		fmt.Println("1. DeepFM 服务已启动: cd python && uvicorn service.deepfm_server:app --host 0.0.0.0 --port 8080")
		fmt.Println("2. 模型已训练: cd python && python train/train_deepfm.py")
		return
	}

	fmt.Printf("推荐结果（用户 ID: %s，使用 Python DeepFM 模型）:\n", rctx.UserID)
	for i, it := range items {
		fmt.Printf("#%d id=%s score=%.4f\n", i, it.ID, it.Score)
		fmt.Printf("  特征示例: item_ctr=%.2f, item_cvr=%.2f, item_price=%.0f\n",
			it.Features["item_ctr"],
			it.Features["item_cvr"],
			it.Features["item_price"],
		)
		fmt.Printf("  Labels: %v\n", it.Labels)
		fmt.Println()
	}
}
