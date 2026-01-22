package rerank

import (
	"context"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// TopNNode 是一个 Top-N 截断节点，用于在排序后截取前 N 个物品。
// 通常在排序（Rank）节点之后使用，用于限制返回结果数量。
//
// 使用场景：
//   - 排序后只返回 Top 10/20/50 个结果
//   - 控制推荐结果数量，提升性能
//   - 配合多样性重排使用
//
// 示例：
//
//	pipeline := &pipeline.Pipeline{
//	    Nodes: []pipeline.Node{
//	        &rank.LRNode{...},        // 排序
//	        &rerank.TopNNode{N: 20},  // 截取 Top 20
//	        &rerank.Diversity{...},   // 多样性重排
//	    },
//	}
type TopNNode struct {
	// N 要保留的物品数量（Top N）
	// 如果 N <= 0，则返回所有物品（不截断）
	// 如果 N > len(items)，则返回所有物品
	N int
}

func (n *TopNNode) Name() string {
	return "rerank.topn"
}

func (n *TopNNode) Kind() pipeline.Kind {
	return pipeline.KindReRank
}

func (n *TopNNode) Process(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	// 如果 N <= 0，不截断，返回所有物品
	if n.N <= 0 {
		return items, nil
	}

	// 如果物品数量小于等于 N，直接返回
	if len(items) <= n.N {
		return items, nil
	}

	// 截取前 N 个物品
	return items[:n.N], nil
}
