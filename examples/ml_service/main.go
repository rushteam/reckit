package main

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/service"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// ========== 1. TensorFlow Serving 示例 ==========
	fmt.Println("=== TensorFlow Serving 示例 ===")

	// 使用 REST API
	tfServiceREST := service.NewTFServingClient(
		"http://localhost:8501",
		"rank_model",
		service.WithTFServingVersion("1"),
		service.WithTFServingTimeout(30*time.Second),
	)
	defer tfServiceREST.Close()

	// 健康检查
	if err := tfServiceREST.Health(ctx); err != nil {
		fmt.Printf("TF Serving 健康检查失败: %v\n", err)
	} else {
		fmt.Println("TF Serving 服务正常")
	}

	// 批量预测
	features := [][]float64{
		{0.1, 0.2, 0.3, 0.4},
		{0.5, 0.6, 0.7, 0.8},
	}
	// 使用 core.MLPredictRequest（符合 DDD）
	resp, err := tfServiceREST.Predict(ctx, &core.MLPredictRequest{
		Instances: features,
	})
	if err != nil {
		fmt.Printf("TF Serving 预测失败: %v\n", err)
	} else {
		fmt.Printf("TF Serving 预测成功，返回 %d 个结果\n", len(resp.Predictions))
		for i, score := range resp.Predictions {
			fmt.Printf("  实例 %d: %.4f\n", i+1, score)
		}
	}

	// 使用 gRPC（推荐）
	tfServiceGRPC := service.NewTFServingClient(
		"localhost:8500",
		"rank_model",
		service.WithTFServingGRPC(),
		service.WithTFServingVersion("1"),
	)
	defer tfServiceGRPC.Close()

	fmt.Println("\n=== 使用 gRPC 协议 ===")
	// 使用 core.MLPredictRequest
	resp, err = tfServiceGRPC.Predict(ctx, &core.MLPredictRequest{
		Instances: features,
	})
	if err != nil {
		fmt.Printf("TF Serving (gRPC) 预测失败: %v\n", err)
	} else {
		fmt.Printf("TF Serving (gRPC) 预测成功\n")
	}

	// ========== 2. 使用工厂方法 ==========
	fmt.Println("\n=== 使用工厂方法 ===")

	// TF Serving 配置
	tfConfig := &service.ServiceConfig{
		Type:        service.ServiceTypeTFServing,
		Endpoint:    "http://localhost:8501",
		ModelName:   "rank_model",
		ModelVersion: "1",
		Timeout:     30,
	}

	mlService, err := service.NewMLService(tfConfig)
	if err != nil {
		fmt.Printf("创建服务失败: %v\n", err)
	} else {
		defer mlService.Close()
		fmt.Println("使用工厂方法创建服务成功")

		// 测试连接
		if err := service.TestConnection(ctx, mlService); err != nil {
			fmt.Printf("测试连接失败: %v\n", err)
		} else {
			fmt.Println("服务连接正常")
		}
	}

	// 注意：ANN（向量检索）应该使用 core.VectorService 接口，而不是 core.MLService
	// 请参考 ext/vector/milvus 或 store.MemoryVectorService 实现向量检索
	fmt.Println("\n注意：向量检索应使用 core.VectorService，请参考向量服务示例")
}
