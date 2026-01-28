package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
	
	// Feast 扩展包
	feasthttp "github.com/rushteam/reckit/ext/feast/http"
	feastgrpc "github.com/rushteam/reckit/ext/feast/grpc"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// 1. 创建 Feast 客户端（扩展包，基础设施层）
	// 方式 A：使用 HTTP 客户端（扩展包）
	// 安装：go get github.com/rushteam/reckit/ext/feast/http
	// feastClient, err := feasthttp.NewClient(
	// 	"http://localhost:6566", // Feast Feature Server 端点
	// 	"my_project",             // 项目名称
	// )
	
	// 方式 B：使用 gRPC 客户端（扩展包，官方 SDK，性能更好，推荐生产环境）
	// 安装：go get github.com/rushteam/reckit/ext/feast/grpc
	feastClient, err := feastgrpc.NewGrpcClient(
		"localhost",  // 主机地址
		6565,         // gRPC 端口（默认 6565）
		"my_project", // 项目名称
	)
	
	if err != nil {
		log.Fatalf("创建 Feast 客户端失败: %v", err)
	}

	// 2. 创建特征映射配置
	mapping := &feasthttp.FeatureMapping{
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

	// 3. 创建适配器（将 Feast 基础设施层接口适配为 feature.FeatureService 领域层接口）
	// 适配器位于扩展包中，这是推荐的使用方式
	adapter := feasthttp.NewFeatureServiceAdapter(feastClient, mapping)
	
	// 关闭适配器（适配器会关闭底层客户端）
	defer func() {
		if err := adapter.Close(context.Background()); err != nil {
			log.Printf("关闭特征服务失败: %v", err)
		}
	}()

	// 4. 创建特征注入节点（使用领域层接口 feature.FeatureService）
	enrichNode := &feature.EnrichNode{
		FeatureService:     adapter, // adapter 实现了 feature.FeatureService 接口
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

	// 10. 演示直接使用 feature.FeatureService（推荐方式）
	fmt.Println("\n=== 使用 feature.FeatureService（领域层接口） ===")

	// 获取用户特征
	userFeatures, err := adapter.GetUserFeatures(ctx, "1001")
	if err != nil {
		log.Printf("获取用户特征失败: %v", err)
	} else {
		fmt.Println("用户特征:")
		for k, v := range userFeatures {
			fmt.Printf("  %s: %v\n", k, v)
		}
	}

	// 批量获取物品特征
	itemFeatures, err := adapter.BatchGetItemFeatures(ctx, []string{"1", "2", "3"})
	if err != nil {
		log.Printf("批量获取物品特征失败: %v", err)
	} else {
		fmt.Println("\n物品特征:")
		for itemID, features := range itemFeatures {
			fmt.Printf("  Item %s: %+v\n", itemID, features)
		}
	}
}
