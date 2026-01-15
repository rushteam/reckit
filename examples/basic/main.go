package main

import (
	"context"
	"fmt"
	"strconv"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/rerank"
	"github.com/rushteam/reckit/store"
)

type enrichDemoFeatures struct{}

func (n *enrichDemoFeatures) Name() string        { return "demo.enrich" }
func (n *enrichDemoFeatures) Kind() pipeline.Kind { return pipeline.KindPostProcess }

func (n *enrichDemoFeatures) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	// 为物品注入基础特征（这些特征会在特征注入节点中被添加 item_ 前缀）
	for i, it := range items {
		it.Features["ctr"] = float64(i+1) * 0.1
		it.Features["cvr"] = float64(6-i) * 0.05
		if i%2 == 0 {
			it.Meta["category"] = "A"
		} else {
			it.Meta["category"] = "B"
		}
	}
	return items, nil
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// 初始化 Store（这里用 MemoryStore 演示，生产环境可用 RedisStore）
	var memStore store.Store = store.NewMemoryStore()
	defer memStore.Close()

	// 演示：将热门物品写入 Store（使用有序集合，按热度分数排序）
	// 生产环境通常由离线任务定期更新
	if kvStore, ok := memStore.(store.KeyValueStore); ok {
		hotKey := "hot:feed"
		hotItems := []struct {
			id    int64
			score float64
		}{
			{1, 100.0}, {2, 95.0}, {3, 90.0}, {4, 85.0}, {5, 80.0},
		}
		for _, item := range hotItems {
			kvStore.ZAdd(ctx, hotKey, item.score, strconv.FormatInt(item.id, 10))
		}
		fmt.Printf("已写入 %d 个热门物品到 Store (key=%s)\n", len(hotItems), hotKey)
	}

	// Demo：直接在内存里构造一个 LR 模型（生产环境通常从文件或远程加载）。
	// 注意：特征注入节点会添加前缀，所以这里使用 item_ 前缀
	lr := &model.LRModel{
		Bias: 0,
		Weights: map[string]float64{
			"item_ctr": 1.2, // 特征注入后会变成 item_ctr
			"item_cvr": 0.8,
			"user_age": 0.5, // 用户特征
		},
	}

	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					// Hot Recall 从 Store 读取热门物品（如果 Store 为空则使用 fallback IDs）
					&recall.Hot{
						Store: memStore,
						Key:   "hot:feed",
					},
				},
				Dedup: true,
			},
			&enrichDemoFeatures{},
			// 特征注入（千人千面）
			&feature.EnrichNode{
				UserFeaturePrefix:  "user_",
				ItemFeaturePrefix:  "item_",
				CrossFeaturePrefix: "cross_",
			},
			&rank.LRNode{Model: lr},
			&rerank.Diversity{LabelKey: "category"},
		},
	}

	rctx := &core.RecommendContext{
		UserID: "42",
		Scene:  "feed",
		Params: map[string]any{"debug": true},
		UserProfile: map[string]any{
			"age":    25,
			"gender": 1, // 1=male, 0=female
		},
		Realtime: map[string]any{
			"hour": time.Now().Hour(),
		},
	}

	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		panic(err)
	}

	for i, it := range items {
		fmt.Printf("#%d id=%s score=%.4f labels=%v\n", i, it.ID, it.Score, it.Labels)
	}
}
