package main

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/examples/feedback"
)

func main() {
	ctx := context.Background()

	// 1. 创建 Kafka 反馈收集器
	collector, err := feedback.NewKafkaCollector(feedback.KafkaCollectorConfig{
		Brokers:       []string{"localhost:9092"},
		Topic:         "feedback-topic",
		BatchSize:     100,
		FlushInterval: 1 * time.Second,
		ClientID:      "reckit-feedback",
		RequiredAcks:  1, // 只需要 leader ACK（性能与可靠性平衡）
		Compression:   "gzip",
		Idempotent:    false, // 根据需求启用
		MaxRetries:    3,
	})
	if err != nil {
		panic(fmt.Sprintf("创建反馈收集器失败: %v", err))
	}
	defer collector.Close()

	// 2. 创建 Pipeline Hook
	feedbackHook := feedback.NewFeedbackHook(collector)

	// 3. 构建 Pipeline
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					&recall.Hot{IDs: []string{"1", "2", "3", "4", "5"}},
				},
				Dedup: true,
			},
		},
		Hooks: []pipeline.PipelineHook{
			feedbackHook, // 添加反馈 Hook
		},
	}

	// 4. 执行推荐
	rctx := &core.RecommendContext{
		UserID: "user_123",
		Scene:  "feed",
		User: &core.UserProfile{
			UserID: "user_123",
			Age:    25,
		},
	}

	fmt.Println("开始执行推荐 Pipeline...")
	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		panic(fmt.Sprintf("Pipeline 执行失败: %v", err))
	}

	fmt.Printf("推荐结果: %d 个物品\n", len(items))
	for i, item := range items {
		fmt.Printf("  %d. ItemID=%s, Score=%.4f\n", i+1, item.ID, item.Score)
	}

	// 5. 模拟用户点击（实际应该在业务代码中调用）
	fmt.Println("\n模拟用户点击...")
	collector.RecordClick(ctx, rctx, items[0].ID, 0)

	// 6. 模拟用户转化
	fmt.Println("模拟用户转化...")
	collector.RecordConversion(ctx, rctx, items[0].ID, map[string]any{
		"amount":   99.0,
		"order_id": "order_123",
	})

	// 等待一段时间，确保数据发送完成
	fmt.Println("\n等待数据发送完成...")
	time.Sleep(2 * time.Second)
	fmt.Println("完成！")
}
