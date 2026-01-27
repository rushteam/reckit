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
	"github.com/rushteam/reckit/pkg/utils"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/store"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 创建用户画像
	userProfile := core.NewUserProfile("1")
	userProfile.Age = 25
	userProfile.Gender = "male"
	userProfile.UpdateInterest("tech", 0.8)
	userProfile.UpdateInterest("game", 0.6)

	// 添加用户行为序列（用于 DIN 模型）
	userProfile.AddRecentClick("1", 10)
	userProfile.AddRecentClick("2", 10)
	userProfile.AddRecentClick("3", 10)

	// 创建 RecommendContext
	rctx := &core.RecommendContext{
		UserID:  "1",
		Scene:   "feed",
		User:    userProfile,
		Labels:  make(map[string]utils.Label),
		Realtime: map[string]any{
			"hour": float64(time.Now().Hour()),
		},
	}

	memStore := store.NewMemoryStore()
	defer memStore.Close(ctx)

	// 特征注入节点
	enrichNode := &feature.EnrichNode{
		UserFeaturePrefix:  "user_",
		ItemFeaturePrefix:  "item_",
		CrossFeaturePrefix: "cross_",
	}

	// ========== 1. DNN 模型 ==========
	fmt.Println("=== 1. DNN 模型 ===")
	dnnModel := model.NewDNNModel([]int{128, 64, 32, 1})
	p1 := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					&recall.Hot{IDs: []string{"1", "2", "3", "4", "5"}},
				},
				Dedup: true,
			},
			&demoItemFeatures{},
			enrichNode,
			&rank.DNNNode{Model: dnnModel},
		},
	}
	items1, _ := p1.Run(ctx, rctx, nil)
	printResults("DNN", items1)

	// ========== 2. Wide&Deep 模型 ==========
	fmt.Println("\n=== 2. Wide&Deep 模型 ===")
	wideDeepModel := model.NewWideDeepModel(
		[]string{"user_age_x_item_ctr", "user_gender_x_item_category"}, // Wide 特征
		[]string{"user_age", "user_gender", "item_ctr", "item_cvr"},      // Deep 特征
		[]int{128, 64, 32, 1}, // Deep 层结构
	)
	// 设置 Wide 权重
	wideDeepModel.WideWeights["user_age_x_item_ctr"] = 0.5
	wideDeepModel.WideWeights["user_gender_x_item_category"] = 0.3
	wideDeepModel.WideBias = 0.1

	p2 := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					&recall.Hot{IDs: []string{"1", "2", "3", "4", "5"}},
				},
				Dedup: true,
			},
			&demoItemFeatures{},
			enrichNode,
			&rank.WideDeepNode{Model: wideDeepModel},
		},
	}
	items2, _ := p2.Run(ctx, rctx, nil)
	printResults("Wide&Deep", items2)

	// ========== 3. DIN 模型 ==========
	fmt.Println("\n=== 3. DIN 模型（行为序列） ===")
	dinModel := model.NewDINModel(32, []int{64, 32}, []int{128, 64, 32, 1})
	// 初始化物品嵌入（简化示例）
	dinModel.ItemEmbeddings["1"] = make([]float64, 32)
	dinModel.ItemEmbeddings["2"] = make([]float64, 32)
	dinModel.ItemEmbeddings["3"] = make([]float64, 32)

	p3 := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					&recall.Hot{IDs: []string{"1", "2", "3", "4", "5"}},
				},
				Dedup: true,
			},
			&demoItemFeatures{},
			enrichNode,
			&rank.DINNode{Model: dinModel, MaxBehaviorSeqLen: 10},
		},
	}
	items3, _ := p3.Run(ctx, rctx, nil)
	printResults("DIN", items3)

	// ========== 4. 两塔模型 ==========
	fmt.Println("\n=== 4. 两塔模型（User Tower + Item Tower） ===")
	twoTowerModel := model.NewTwoTowerModel(
		[]int{128, 64, 32}, // User Tower
		[]int{128, 64, 32}, // Item Tower
		32,                 // Embedding 维度
	)
	twoTowerModel.SimilarityType = "dot" // 使用内积

	p4 := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					&recall.Hot{IDs: []string{"1", "2", "3", "4", "5"}},
				},
				Dedup: true,
			},
			&demoItemFeatures{},
			enrichNode,
			&rank.TwoTowerNode{Model: twoTowerModel},
		},
	}
	items4, _ := p4.Run(ctx, rctx, nil)
	printResults("TwoTower", items4)
}

// demoItemFeatures 为物品添加演示特征
type demoItemFeatures struct{}

func (n *demoItemFeatures) Name() string {
	return "demo.item_features"
}

func (n *demoItemFeatures) Kind() pipeline.Kind {
	return pipeline.KindPostProcess
}

func (n *demoItemFeatures) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	for _, item := range items {
		if item == nil {
			continue
		}
		// 添加物品特征
		// 将 string ID 转换为 float64（如果 ID 是数字字符串）
		var itemIDFloat float64
		if id, err := strconv.ParseFloat(item.ID, 64); err == nil {
			itemIDFloat = id
		} else {
			// 对于非数字 ID，使用哈希值
			hash := 0.0
			for _, c := range item.ID {
				hash = hash*31.0 + float64(c)
			}
			itemIDFloat = hash
		}
		item.Features["item_ctr"] = 0.15 + itemIDFloat*0.01
		item.Features["item_cvr"] = 0.08 + itemIDFloat*0.005
		item.Features["item_price"] = itemIDFloat * 10.0
		item.PutLabel("category", utils.Label{Value: "tech", Source: "demo"})
	}
	return items, nil
}

func printResults(modelName string, items []*core.Item) {
	fmt.Printf("模型: %s\n", modelName)
	for i, item := range items {
		if item == nil {
			continue
		}
		fmt.Printf("  %d. 物品 %s (分数: %.4f)", i+1, item.ID, item.Score)
		if item.Labels != nil {
			if rankType, ok := item.Labels["rank_type"]; ok {
				fmt.Printf(" [%s]", rankType.Value)
			}
		}
		fmt.Println()
	}
}
