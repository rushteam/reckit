package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feast"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// 1. 创建 Feast 客户端
	// 方式 A：使用 HTTP 客户端（自定义实现，支持完整功能）
	// feastClient, err := feast.NewHTTPClient(
	// 	"http://localhost:6566", // Feast Feature Server 端点
	// 	"my_project",             // 项目名称
	// )
	
	// 方式 B：使用 gRPC 客户端（官方 SDK，性能更好，推荐生产环境）
	feastClient, err := feast.NewGrpcClient(
		"localhost",  // 主机地址
		6565,         // gRPC 端口（默认 6565）
		"my_project", // 项目名称
	)
	
	// 方式 C：使用工厂模式（自动选择）
	// factory := &feast.DefaultClientFactory{}
	// feastClient, err := factory.NewClient(
	// 	ctx,
	// 	"localhost:6565",
	// 	"my_project",
	// 	feast.WithGRPC(), // 或 feast.WithHTTP()
	// )
	
	if err != nil {
		log.Fatalf("创建 Feast 客户端失败: %v", err)
	}
	defer feastClient.Close()

	// 2. 创建特征映射配置
	mapping := &feast.FeatureMapping{
		UserFeatures: []string{
			"user_stats:age",
			"user_stats:gender",
			"user_stats:city",
		},
		ItemFeatures: []string{
			"item_stats:price",
			"item_stats:category",
			"item_stats:rating",
		},
		RealtimeFeatures: []string{
			"interaction:click_count",
			"interaction:view_count",
		},
		UserEntityKey: "user_id",
		ItemEntityKey: "item_id",
	}

	// 3. 创建适配器（将 Feast Client 适配为 FeatureService）
	adapter := feast.NewFeatureServiceAdapter(feastClient, mapping)

	// 4. 创建特征注入节点
	enrichNode := &feature.EnrichNode{
		FeatureService:     adapter,
		UserFeaturePrefix:  "user_",
		ItemFeaturePrefix:  "item_",
		CrossFeaturePrefix: "cross_",
	}

	// 5. 创建排序模型
	lr := &rank.LRNode{
		Model: &model.LRModel{
			Bias: 0,
			Weights: map[string]float64{
				"item_price":  -0.1,
				"item_rating": 1.5,
				"user_age":    0.3,
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
			// 特征注入（使用 Feast）
			enrichNode,
			// 排序
			lr,
		},
	}

	// 7. 创建用户上下文
	rctx := &core.RecommendContext{
		UserID: "1001",
		Scene:  "feed",
	}

	// 8. 运行 Pipeline
	fmt.Println("开始运行 Pipeline（使用 Feast Feature Store）...")
	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		log.Fatalf("Pipeline 运行失败: %v", err)
	}

	// 9. 输出结果
	fmt.Printf("\n推荐结果（共 %d 个物品）:\n", len(items))
	for i, item := range items {
		fmt.Printf("%d. ItemID=%s, Score=%.4f\n", i+1, item.ID, item.Score)
	}

	// 10. 演示直接使用 Feast 客户端
	fmt.Println("\n=== 直接使用 Feast 客户端 ===")

	// 获取在线特征
	onlineReq := &feast.GetOnlineFeaturesRequest{
		Features: []string{
			"user_stats:age",
			"user_stats:gender",
		},
		EntityRows: []map[string]interface{}{
			{"user_id": 1001},
			{"user_id": 1002},
		},
	}

	onlineResp, err := feastClient.GetOnlineFeatures(ctx, onlineReq)
	if err != nil {
		log.Printf("获取在线特征失败: %v", err)
	} else {
		fmt.Println("在线特征:")
		for i, fv := range onlineResp.FeatureVectors {
			fmt.Printf("  Entity %d: %+v\n", i+1, fv.Values)
		}
	}

	// 列出所有特征
	features, err := feastClient.ListFeatures(ctx)
	if err != nil {
		log.Printf("列出特征失败: %v", err)
	} else {
		fmt.Printf("\n可用特征（共 %d 个）:\n", len(features))
		for _, f := range features {
			fmt.Printf("  - %s (%s)\n", f.Name, f.ValueType)
		}
	}

	// 获取特征服务信息
	info, err := feastClient.GetFeatureService(ctx)
	if err != nil {
		log.Printf("获取特征服务信息失败: %v", err)
	} else {
		fmt.Printf("\n特征服务信息:\n")
		fmt.Printf("  Endpoint: %s\n", info.Endpoint)
		fmt.Printf("  Project: %s\n", info.Project)
		fmt.Printf("  Online Store: %s\n", info.OnlineStore)
		fmt.Printf("  Offline Store: %s\n", info.OfflineStore)
	}
}
