package main

import (
	"context"
	"fmt"
	"time"

	"reckit/config"
	"reckit/core"
	"reckit/pipeline"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 从 YAML 配置文件加载 Pipeline
	// 从项目根目录运行时，使用相对路径
	cfg, err := pipeline.LoadFromYAML("examples/config/pipeline.example.yaml")
	if err != nil {
		fmt.Printf("加载配置失败: %v\n", err)
		return
	}

	// 使用默认工厂构建 Pipeline
	factory := config.DefaultFactory()
	p, err := cfg.BuildPipeline(factory)
	if err != nil {
		fmt.Printf("构建 Pipeline 失败: %v\n", err)
		return
	}

	// 创建推荐上下文
	rctx := &core.RecommendContext{
		UserID: 42,
		Scene:  "feed",
		Params: map[string]any{"debug": true},
		Realtime: map[string]any{
			"hour": time.Now().Hour(),
		},
	}

	// 运行 Pipeline
	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		fmt.Printf("运行 Pipeline 失败: %v\n", err)
		return
	}

	// 输出结果
	fmt.Printf("Pipeline 执行成功，返回 %d 个物品:\n", len(items))
	for i, it := range items {
		fmt.Printf("#%d id=%d score=%.4f labels=%v\n", i, it.ID, it.Score, it.Labels)
	}
}
