package recall

import (
	"context"

	"reckit/core"
)

// Source 表示一个可复用的召回源（热门/CF/内容/ANN/...）。
// 你可以把它理解为“可并发 fan-out 的策略单元”。
type Source interface {
	Name() string
	Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error)
}
