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

// BatchFilter 是批量过滤器接口，适用于需要批量查询外部服务的过滤场景
// （如批量查 Redis/DB 判断已曝光）。一次调用完成所有 item 的过滤，
// 返回保留的 item 列表，避免逐条查询的 N+1 问题。
//
// FilterNode 会优先检测 Filter 是否同时实现了 BatchFilter，
// 如果实现则调用 FilterBatch，否则降级为逐条调用 ShouldFilter。
type BatchFilter interface {
	Filter

	// FilterBatch 批量过滤，返回应保留的 item 列表。
	// 实现应自行处理内部错误（如部分查询失败），对无法判断的 item 建议保留（宁可漏过，不可误杀）。
	FilterBatch(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) ([]*core.Item, error)
}
