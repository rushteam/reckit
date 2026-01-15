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

// demoItemFeatures 为物品注入基础特征（示例）
type demoItemFeatures struct{}

func (n *demoItemFeatures) Name() string        { return "demo.item_features" }
func (n *demoItemFeatures) Kind() pipeline.Kind { return pipeline.KindPostProcess }

func (n *demoItemFeatures) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	// 为物品注入基础特征
	for i, it := range items {
		it.Features["ctr"] = float64(i+1) * 0.1
		it.Features["cvr"] = float64(6-i) * 0.05
		it.Features["price"] = float64(100 - i*10)
		if i%2 == 0 {
			it.Meta["category"] = "A"
		} else {
			it.Meta["category"] = "B"
		}
	}
	return items, nil
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 初始化 Store
	memStore := store.NewMemoryStore()
	defer memStore.Close()

	// 创建特征注入节点（千人千面核心）
	enrichNode := &feature.EnrichNode{
		UserFeaturePrefix:  "user_",
		ItemFeaturePrefix:  "item_",
		CrossFeaturePrefix: "cross_",
	}

	// 创建 LR 模型（支持用户特征、物品特征、交叉特征）
	lr := &model.LRModel{
		Bias: 0,
		Weights: map[string]float64{
			// 物品特征（需要先通过特征注入节点注入）
			"item_ctr": 1.2,
			"item_cvr": 0.8,
			// 用户特征
			"user_age":    0.5,
			"user_gender": 0.3,
			// 交叉特征
			"cross_age_x_ctr": 0.2, // 年龄 × CTR 的交叉特征
		},
	}

	// 构建 Pipeline（支持千人千面）
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			// 召回
			&recall.Fanout{
				Sources: []recall.Source{
					&recall.Hot{IDs: []string{"1", "2", "3", "4", "5"}},
				},
				Dedup: true,
			},
			// 过滤（个性化过滤）
			&filter.FilterNode{
				Filters: []filter.Filter{
					filter.NewUserBlockFilter(nil, "user:block"),
					filter.NewExposedFilter(nil, "user:exposed", 7*24*3600),
				},
			},
			// 注入物品基础特征（示例：在实际场景中，这些特征可能来自特征服务）
			&demoItemFeatures{},
			// 特征注入（千人千面关键步骤：将用户特征、物品特征、交叉特征组合）
			enrichNode,
			// 排序（使用个性化特征）
			&rank.LRNode{Model: lr},
			// 重排
			&rerank.Diversity{LabelKey: "category"},
		},
	}

	// 创建用户上下文（包含用户画像）
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
	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		panic(err)
	}

	fmt.Printf("千人千面推荐结果（用户 ID: %s）:\n", rctx.UserID)
	for i, it := range items {
		fmt.Printf("#%d id=%s score=%.4f\n", i, it.ID, it.Score)
		fmt.Printf("  特征: %v\n", it.Features)
		fmt.Printf("  Labels: %v\n", it.Labels)
		fmt.Println()
	}
}
