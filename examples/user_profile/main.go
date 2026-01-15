package main

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/rerank"
	"github.com/rushteam/reckit/store"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 1. 创建用户画像
	userProfile := core.NewUserProfile("1")
	userProfile.Age = 25
	userProfile.Gender = "male"
	userProfile.Location = "beijing"

	// 设置长期兴趣
	userProfile.UpdateInterest("tech", 0.8)
	userProfile.UpdateInterest("game", 0.6)
	userProfile.UpdateInterest("news", 0.5)

	// 设置实验桶
	userProfile.SetBucket("diversity", "strong")
	userProfile.SetBucket("recall", "v2")

	// 2. 创建 RecommendContext（包含用户画像）
	rctx := &core.RecommendContext{
		UserID:  "1",
		Scene:   "feed",
		User:    userProfile,
		Labels:  make(map[string]utils.Label),
		Realtime: map[string]any{
			"hour": float64(time.Now().Hour()),
		},
	}

	// 设置用户级标签
	rctx.PutLabel("user_type", utils.Label{Value: "active", Source: "system"})
	rctx.PutLabel("price_sensitive", utils.Label{Value: "false", Source: "system"})

	// 3. 创建 Pipeline（用户画像驱动）
	memStore := store.NewMemoryStore()
	defer memStore.Close()

	// 创建带类别标签的热门召回
	hotRecall := &recallWithCategory{}

	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			// 召回（用户画像驱动）
			&recall.Fanout{
				Sources: []recall.Source{
					hotRecall,
				},
				Dedup: true,
			},
			// 用户画像驱动的召回增强
			&userProfileDrivenRecall{},
			// 排序（用户画像驱动）
			&rank.LRNode{
				Model: &model.LRModel{
					Bias: 0,
					Weights: map[string]float64{
						"item_ctr": 1.2,
						"item_cvr": 0.8,
					},
				},
			},
			// 用户画像驱动的排序增强
			&userProfileDrivenRank{},
			// 重排（用户画像驱动）
			&rerank.Diversity{LabelKey: "category"},
			// 用户画像驱动的重排
			&userProfileDrivenRerank{},
		},
	}

	// 4. 运行 Pipeline
	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		panic(err)
	}

	// 5. 输出结果
	fmt.Println("=== 用户画像驱动的推荐结果 ===")
	for i, item := range items {
		fmt.Printf("%d. 物品 %s (分数: %.4f)\n", i+1, item.ID, item.Score)
		// 输出相关标签
		if item.Labels != nil {
			for k, v := range item.Labels {
				if k == "user_interest" || k == "user_interest_boost" || k == "diversity_boost" {
					fmt.Printf("   - %s: %s\n", k, v.Value)
				}
			}
		}
	}

	// 6. 模拟用户点击，回写 Label
	fmt.Println("\n=== 用户点击回写 ===")
	clickedItemID := items[0].ID
	rctx.PutLabel(fmt.Sprintf("user.click.%s", clickedItemID), utils.Label{
		Value:  "1",
		Source: "feedback",
	})
	userProfile.AddRecentClick(clickedItemID, 100)
	fmt.Printf("用户点击物品 %s，已回写到 UserProfile\n", clickedItemID)

	// 7. Online Learning：根据点击更新兴趣
	if category, ok := items[0].Labels["category"]; ok {
		currentWeight := userProfile.GetInterestWeight(category.Value)
		newWeight := currentWeight + 0.1 // 简单示例：点击后增加 0.1
		if newWeight > 1.0 {
			newWeight = 1.0
		}
		userProfile.UpdateInterest(category.Value, newWeight)
		fmt.Printf("更新兴趣 %s: %.2f -> %.2f\n", category.Value, currentWeight, newWeight)
	}
}

// recallWithCategory 是带类别标签的召回源（用于演示）。
type recallWithCategory struct{}

func (r *recallWithCategory) Name() string {
	return "recall.hot_with_category"
}

func (r *recallWithCategory) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	ids := []string{"1", "2", "3", "4", "5"}
	out := make([]*core.Item, 0, len(ids))
	categories := []string{"tech", "game", "news", "music", "sports"}
	for i, id := range ids {
		item := core.NewItem(id)
		category := categories[i%len(categories)]
		item.PutLabel("category", utils.Label{Value: category, Source: "demo"})
		out = append(out, item)
	}
	return out, nil
}

// userProfileDrivenRecall 是用户画像驱动的召回增强 Node。
type userProfileDrivenRecall struct{}

func (n *userProfileDrivenRecall) Name() string {
	return "recall.user_profile_driven"
}

func (n *userProfileDrivenRecall) Kind() pipeline.Kind {
	return pipeline.KindRecall
}

func (n *userProfileDrivenRecall) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if rctx.User == nil || len(items) == 0 {
		return items, nil
	}

	// 根据用户兴趣对物品进行加权
	for _, item := range items {
		if item == nil {
			continue
		}

		// 检查物品类别是否匹配用户兴趣
		if category, ok := item.Labels["category"]; ok {
			if rctx.User.HasInterest(category.Value, 0.5) {
				// 用户偏好放大
				weight := rctx.User.GetInterestWeight(category.Value)
				item.Score += weight * 0.5
				item.PutLabel("user_interest", utils.Label{
					Value:  category.Value,
					Source: n.Name(),
				})
			}
		}
	}

	return items, nil
}

// userProfileDrivenRank 是用户画像驱动的排序增强 Node。
type userProfileDrivenRank struct{}

func (n *userProfileDrivenRank) Name() string {
	return "rank.user_profile_driven"
}

func (n *userProfileDrivenRank) Kind() pipeline.Kind {
	return pipeline.KindRank
}

func (n *userProfileDrivenRank) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if rctx.User == nil || len(items) == 0 {
		return items, nil
	}

	// 根据用户兴趣对排序分数进行加权
	for _, item := range items {
		if item == nil {
			continue
		}

		// 用户兴趣加权
		if category, ok := item.Labels["category"]; ok {
			if weight := rctx.User.GetInterestWeight(category.Value); weight > 0 {
				item.Score *= (1 + weight)
				item.PutLabel("user_interest_boost", utils.Label{
					Value:  category.Value,
					Source: n.Name(),
				})
			}
		}
	}

	return items, nil
}

// userProfileDrivenRerank 是用户画像驱动的重排 Node。
type userProfileDrivenRerank struct{}

func (n *userProfileDrivenRerank) Name() string {
	return "rerank.user_profile_driven"
}

func (n *userProfileDrivenRerank) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *userProfileDrivenRerank) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if rctx.User == nil || len(items) == 0 {
		return items, nil
	}

	// 根据实验桶调整多样性
	if rctx.User.GetBucket("diversity") == "strong" {
		for _, item := range items {
			if item == nil {
				continue
			}
			// 多样性惩罚（示例）
			item.Score *= 0.7
			item.PutLabel("diversity_boost", utils.Label{
				Value:  "strong",
				Source: n.Name(),
			})
		}
	}

	return items, nil
}
