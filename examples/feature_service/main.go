package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"reckit/core"
	"reckit/feature"
	"reckit/model"
	"reckit/pipeline"
	"reckit/rank"
	"reckit/recall"
	"reckit/store"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 1. 初始化 Store
	var s store.Store
	redisStore, err := store.NewRedisStore("localhost:6379", 0)
	if err != nil {
		log.Printf("Redis 连接失败，使用内存 Store: %v", err)
		s = store.NewMemoryStore()
	} else {
		s = redisStore
	}
	defer s.Close()

	// 2. 准备特征数据（示例：实际场景中由离线任务或实时任务更新）
	prepareFeatureData(ctx, s)

	// 3. 创建特征服务（使用工厂模式）
	factory := feature.NewFeatureServiceFactory()
	featureService := factory.CreateFull(
		s,
		10000,         // 缓存大小
		5*time.Minute, // 缓存 TTL
		1000,          // 监控样本数
	)
	defer featureService.Close()

	// 4. 创建特征注入节点（使用特征服务）
	enrichNode := &feature.EnrichNode{
		FeatureService:     featureService, // 使用特征服务
		UserFeaturePrefix:  "user_",
		ItemFeaturePrefix:  "item_",
		CrossFeaturePrefix: "cross_",
	}

	// 5. 创建排序模型
	lr := &rank.LRNode{
		Model: &model.LRModel{
			Bias: 0,
			Weights: map[string]float64{
				"item_ctr": 1.2,
				"item_cvr": 0.8,
				"user_age": 0.5,
			},
		},
	}

	// 6. 构建 Pipeline
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			// 召回
			&recall.Fanout{
				Sources: []recall.Source{
					&recall.Hot{IDs: []string{"1", "2", "3", "4", "5"}},
				},
				Dedup: true,
			},
			// 特征注入（使用特征服务）
			enrichNode,
			// 排序
			lr,
		},
	}

	// 7. 创建用户上下文
	rctx := &core.RecommendContext{
		UserID: "42",
		Scene:  "feed",
		UserProfile: map[string]any{
			"age":    25.0,
			"gender": 1.0,
			"city":   "beijing",
		},
	}

	// 8. 运行 Pipeline
	fmt.Println("开始运行 Pipeline（使用特征服务）...")
	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		log.Fatalf("Pipeline 运行失败: %v", err)
	}

	// 9. 输出结果
	fmt.Printf("\n推荐结果（共 %d 个物品）:\n", len(items))
	for i, item := range items {
		if item == nil {
			continue
		}
		fmt.Printf("\n[%d] Item ID: %s, Score: %.4f\n", i+1, item.ID, item.Score)
		fmt.Printf("    特征数量: %d\n", len(item.Features))
		fmt.Printf("    用户特征: ")
		for k, v := range item.Features {
			if len(k) > 5 && k[:5] == "user_" {
				fmt.Printf("%s=%.2f ", k, v)
			}
		}
		fmt.Printf("\n    物品特征: ")
		for k, v := range item.Features {
			if len(k) > 5 && k[:5] == "item_" {
				fmt.Printf("%s=%.2f ", k, v)
			}
		}
		fmt.Printf("\n    交叉特征: ")
		for k, v := range item.Features {
			if len(k) > 6 && k[:6] == "cross_" {
				fmt.Printf("%s=%.2f ", k, v)
			}
		}
		fmt.Println()
	}

	// 10. 演示特征服务直接使用
	fmt.Println("\n=== 直接使用特征服务 ===")

	// 获取用户特征
	userFeatures, err := featureService.GetUserFeatures(ctx, "42")
	if err != nil {
		log.Printf("获取用户特征失败: %v", err)
	} else {
		fmt.Printf("用户 42 的特征: %v\n", userFeatures)
	}

	// 批量获取物品特征
	itemIDs := []string{"1", "2", "3"}
	itemFeatures, err := featureService.BatchGetItemFeatures(ctx, itemIDs)
	if err != nil {
		log.Printf("批量获取物品特征失败: %v", err)
	} else {
		fmt.Printf("物品特征: %v\n", itemFeatures)
	}

	// 获取实时特征
	realtimeFeatures, err := featureService.GetRealtimeFeatures(ctx, "42", "1")
	if err != nil {
		log.Printf("获取实时特征失败: %v", err)
	} else {
		fmt.Printf("用户 42 对物品 1 的实时特征: %v\n", realtimeFeatures)
	}
}

// prepareFeatureData 准备特征数据（示例）
func prepareFeatureData(ctx context.Context, store store.Store) {
	// 用户特征
	userFeatures := map[string]float64{
		"age":    25.0,
		"gender": 1.0,
		"city":   1.0, // beijing
	}
	userData, _ := json.Marshal(userFeatures)
	store.Set(ctx, "user:features:42", userData, 3600)

	// 物品特征
	itemFeaturesMap := map[int64]map[string]float64{
		1: {"ctr": 0.15, "cvr": 0.08, "price": 99.0},
		2: {"ctr": 0.12, "cvr": 0.05, "price": 150.0},
		3: {"ctr": 0.08, "cvr": 0.03, "price": 200.0},
		4: {"ctr": 0.20, "cvr": 0.10, "price": 50.0},
		5: {"ctr": 0.18, "cvr": 0.09, "price": 80.0},
	}

	for itemID, features := range itemFeaturesMap {
		itemData, _ := json.Marshal(features)
		key := fmt.Sprintf("item:features:%d", itemID)
		store.Set(ctx, key, itemData, 3600)
	}

	fmt.Println("特征数据已准备完成")
}
