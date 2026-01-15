package filter

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// Filter 是过滤器的抽象接口，用于判断一个 Item 是否应该被过滤掉。
// 返回 true 表示应该过滤（移除），false 表示保留。
type Filter interface {
	// Name 返回过滤器名称
	Name() string

	// ShouldFilter 判断 item 是否应该被过滤
	ShouldFilter(ctx context.Context, rctx *core.RecommendContext, item *core.Item) (bool, error)
}
