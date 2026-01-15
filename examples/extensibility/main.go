package main

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/recall"
	"github.com/rushteam/reckit/store"
)

// 示例：自定义合并策略
type CustomMergeStrategy struct {
	// 自定义逻辑：只保留分数最高的物品
}

func (s *CustomMergeStrategy) Merge(items []*core.Item, dedup bool) []*core.Item {
	if !dedup {
		return items
	}
	
	// 按 ID 分组，保留分数最高的
	bestItems := make(map[string]*core.Item)
	for _, item := range items {
		if item == nil {
			continue
		}
		if existing, ok := bestItems[item.ID]; ok {
			if item.Score > existing.Score {
				bestItems[item.ID] = item
			}
		} else {
			bestItems[item.ID] = item
		}
	}
	
	// 转换为列表
	result := make([]*core.Item, 0, len(bestItems))
	for _, item := range bestItems {
		result = append(result, item)
	}
	return result
}

// 示例：Pipeline Hook - 日志记录
type LoggingHook struct{}

func (h *LoggingHook) BeforeNode(ctx context.Context, rctx *core.RecommendContext, node pipeline.Node, items []*core.Item) ([]*core.Item, error) {
	fmt.Printf("[Hook] Before %s: %d items\n", node.Name(), len(items))
	return items, nil
}

func (h *LoggingHook) AfterNode(ctx context.Context, rctx *core.RecommendContext, node pipeline.Node, items []*core.Item, err error) ([]*core.Item, error) {
	if err != nil {
		fmt.Printf("[Hook] After %s: error=%v\n", node.Name(), err)
	} else {
		fmt.Printf("[Hook] After %s: %d items\n", node.Name(), len(items))
	}
	return items, err
}

// 示例：Pipeline Hook - 性能监控
type MetricsHook struct {
	startTime time.Time
}

func (h *MetricsHook) BeforeNode(ctx context.Context, rctx *core.RecommendContext, node pipeline.Node, items []*core.Item) ([]*core.Item, error) {
	h.startTime = time.Now()
	return items, nil
}

func (h *MetricsHook) AfterNode(ctx context.Context, rctx *core.RecommendContext, node pipeline.Node, items []*core.Item, err error) ([]*core.Item, error) {
	duration := time.Since(h.startTime)
	fmt.Printf("[Metrics] %s took %v\n", node.Name(), duration)
	return items, err
}

// 示例：自定义 Node
type CustomNode struct{}

func (n *CustomNode) Name() string {
	return "custom.node"
}

func (n *CustomNode) Kind() pipeline.Kind {
	return pipeline.KindPostProcess
}

func (n *CustomNode) Process(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	// 自定义处理逻辑
	for _, item := range items {
		if item != nil {
			item.Score *= 1.1 // 示例：给所有物品加分
		}
	}
	return items, nil
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// 初始化 Store
	memStore := store.NewMemoryStore()
	defer memStore.Close()

	// 1. 使用自定义合并策略
	fmt.Println("=== 示例 1: 自定义合并策略 ===")
	fanout := &recall.Fanout{
		Sources: []recall.Source{
			&recall.Hot{IDs: []string{"1", "2", "3"}},
		},
		Dedup:         true,
		MergeStrategy: &CustomMergeStrategy{}, // 使用自定义策略
	}

	// 2. 使用 Pipeline Hook
	fmt.Println("\n=== 示例 2: Pipeline Hook ===")
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			fanout,
			&CustomNode{},
		},
		Hooks: []pipeline.PipelineHook{
			&LoggingHook{},
			&MetricsHook{},
		},
	}

	rctx := &core.RecommendContext{
		UserID: "1",
		Scene:  "feed",
	}

	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("\n最终结果: %d 个物品\n", len(items))
	for i, item := range items {
		fmt.Printf("  %d. Item %s, Score: %.2f\n", i+1, item.ID, item.Score)
	}

	// 3. 动态注册自定义 Node
	fmt.Println("\n=== 示例 3: 动态注册 Node ===")
	factory := pipeline.NewNodeFactory()
	
	// 注册自定义 Node
	factory.Register("custom.node", func(config map[string]interface{}) (pipeline.Node, error) {
		return &CustomNode{}, nil
	})

	// 查看已注册的类型
	types := factory.ListRegisteredTypes()
	fmt.Printf("已注册的 Node 类型: %v\n", types)

	// 使用工厂构建 Node
	customNode, err := factory.Build("custom.node", nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("成功构建 Node: %s\n", customNode.Name())
	}
}
