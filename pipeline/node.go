package pipeline

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// Kind 用于标记 Node 类型，方便观测/治理/编排（例如按阶段打点）。
type Kind string

const (
	KindRecall      Kind = "recall"      // 召回阶段：生成候选集
	KindFilter      Kind = "filter"      // 过滤阶段：剔除不符合约束的候选
	KindRank        Kind = "rank"        // 排序阶段：对候选打分并排序
	KindReRank      Kind = "rerank"      // 重排阶段：在排序结果上做多样性/业务调优
	KindPostProcess Kind = "postprocess" // 后处理阶段：补充特征或最终结果修饰
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
