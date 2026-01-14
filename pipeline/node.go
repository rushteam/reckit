package pipeline

import (
	"context"

	"reckit/core"
)

// Kind 用于标记 Node 类型，方便观测/治理/编排（例如按阶段打点）。
type Kind string

const (
	KindRecall      Kind = "recall"
	KindFilter      Kind = "filter"
	KindRank        Kind = "rank"
	KindReRank      Kind = "rerank"
	KindPostProcess Kind = "postprocess"
)

// Node 是 Pipeline 的最小可扩展单元。
// 统一采用“输入 items -> 输出 items”的形态，方便 Recall 生成、Filter 截断、ReRank 重排等操作。
type Node interface {
	Name() string
	Kind() Kind

	Process(
		ctx context.Context,
		rctx *core.RecommendContext,
		items []*core.Item,
	) ([]*core.Item, error)
}
