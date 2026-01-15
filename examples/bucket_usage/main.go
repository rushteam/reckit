package main

import (
	"context"
	"fmt"
	"time"

	"reckit/core"
	"reckit/pipeline"
	"reckit/pkg/utils"
	"reckit/recall"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// ========== SetBucket 使用示例 ==========

	// 1. 创建用户画像并设置实验桶
	userProfile := core.NewUserProfile(1)

	// 设置实验桶：用于 A/B 测试和策略切换
	// key: 实验名称，value: 实验组/策略版本
	userProfile.SetBucket("diversity", "strong")    // 多样性策略：强多样性
	userProfile.SetBucket("recall", "v2")            // 召回策略：版本2
	userProfile.SetBucket("rank", "deep_model")       // 排序策略：深度模型
	userProfile.SetBucket("rerank", "diversity_v1")   // 重排策略：多样性版本1

	// 2. 在 Node 中使用 GetBucket 获取实验桶值
	rctx := &core.RecommendContext{
		UserID: 1,
		Scene:  "feed",
		User:   userProfile,
	}

	// 示例：根据实验桶调整策略
	fmt.Println("=== 实验桶使用示例 ===")
	fmt.Printf("多样性策略: %s\n", rctx.User.GetBucket("diversity"))
	fmt.Printf("召回策略: %s\n", rctx.User.GetBucket("recall"))
	fmt.Printf("排序策略: %s\n", rctx.User.GetBucket("rank"))
	fmt.Printf("重排策略: %s\n", rctx.User.GetBucket("rerank"))

	// 3. 创建 Pipeline，Node 中根据实验桶调整行为
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			// 召回阶段：根据实验桶选择不同的召回源
			&recall.Fanout{
				Sources: []recall.Source{
					&recallWithBucket{},
				},
				Dedup: true,
			},
			// 排序阶段：根据实验桶选择不同的排序策略
			&rankWithBucket{},
			// 重排阶段：根据实验桶调整多样性
			&rerankWithBucket{},
		},
	}

	// 运行 Pipeline
	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		panic(err)
	}

	fmt.Println("\n=== 推荐结果 ===")
	for i, item := range items {
		fmt.Printf("%d. 物品 %d (分数: %.4f)\n", i+1, item.ID, item.Score)
		if item.Labels != nil {
			for k, v := range item.Labels {
				if k == "bucket_strategy" {
					fmt.Printf("   - 策略: %s\n", v.Value)
				}
			}
		}
	}
}

// recallWithBucket 根据实验桶选择不同的召回策略
type recallWithBucket struct{}

func (r *recallWithBucket) Name() string {
	return "recall.bucket_driven"
}

func (r *recallWithBucket) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	// 根据实验桶选择召回策略
	recallVersion := rctx.User.GetBucket("recall")
	
	var items []*core.Item
	switch recallVersion {
	case "v2":
		// 使用新版本召回：更多个性化
		items = []*core.Item{
			core.NewItem(1),
			core.NewItem(2),
			core.NewItem(3),
		}
		fmt.Println("使用召回策略 v2：个性化召回")
	case "v1":
		// 使用旧版本召回：热门召回
		items = []*core.Item{
			core.NewItem(4),
			core.NewItem(5),
		}
		fmt.Println("使用召回策略 v1：热门召回")
	default:
		// 默认策略
		items = []*core.Item{
			core.NewItem(1),
			core.NewItem(2),
		}
		fmt.Println("使用默认召回策略")
	}

	// 为物品打标签，记录使用的策略
	for _, item := range items {
		item.PutLabel("bucket_strategy", utils.Label{
			Value:  fmt.Sprintf("recall_%s", recallVersion),
			Source: r.Name(),
		})
	}

	return items, nil
}

// rankWithBucket 根据实验桶选择不同的排序策略
type rankWithBucket struct{}

func (r *rankWithBucket) Name() string {
	return "rank.bucket_driven"
}

func (r *rankWithBucket) Kind() pipeline.Kind {
	return pipeline.KindRank
}

func (r *rankWithBucket) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	// 根据实验桶选择排序策略
	rankStrategy := rctx.User.GetBucket("rank")

	for _, item := range items {
		switch rankStrategy {
		case "deep_model":
			// 使用深度模型排序
			item.Score = 0.9
			item.PutLabel("bucket_strategy", utils.Label{
				Value:  "rank_deep_model",
				Source: r.Name(),
			})
		case "lr_model":
			// 使用 LR 模型排序
			item.Score = 0.7
			item.PutLabel("bucket_strategy", utils.Label{
				Value:  "rank_lr_model",
				Source: r.Name(),
			})
		default:
			// 默认排序
			item.Score = 0.5
			item.PutLabel("bucket_strategy", utils.Label{
				Value:  "rank_default",
				Source: r.Name(),
			})
		}
	}

	return items, nil
}

// rerankWithBucket 根据实验桶调整重排策略
type rerankWithBucket struct{}

func (r *rerankWithBucket) Name() string {
	return "rerank.bucket_driven"
}

func (r *rerankWithBucket) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (r *rerankWithBucket) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	// 根据实验桶调整多样性
	diversityStrategy := rctx.User.GetBucket("diversity")

	switch diversityStrategy {
	case "strong":
		// 强多样性：降低相似物品的分数
		for _, item := range items {
			item.Score *= 0.7
			item.PutLabel("bucket_strategy", utils.Label{
				Value:  "rerank_diversity_strong",
				Source: r.Name(),
			})
		}
		fmt.Println("使用强多样性策略")
	case "weak":
		// 弱多样性：保持原分数
		for _, item := range items {
			item.PutLabel("bucket_strategy", utils.Label{
				Value:  "rerank_diversity_weak",
				Source: r.Name(),
			})
		}
		fmt.Println("使用弱多样性策略")
	default:
		// 默认策略
		fmt.Println("使用默认重排策略")
	}

	return items, nil
}
