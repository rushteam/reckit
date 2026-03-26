package filter

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// DedupByFieldFilter 按指定字段去重（保留每个值的首条）。
// 字段查找顺序与 Item.GetValue 一致：Labels > Meta > Features。
// 典型场景：同一文章的多个变体只保留一个。
type DedupByFieldFilter struct {
	FieldKey string
}

func (f *DedupByFieldFilter) Name() string { return "filter.dedup_field" }

// FilterBatch 实现 BatchFilter 接口。
func (f *DedupByFieldFilter) FilterBatch(
	_ context.Context,
	_ *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if f.FieldKey == "" {
		return items, nil
	}
	seen := make(map[string]bool, len(items))
	out := make([]*core.Item, 0, len(items))
	for _, it := range items {
		if it == nil {
			continue
		}
		v, ok := it.GetValue(f.FieldKey)
		if !ok || v == "" {
			out = append(out, it)
			continue
		}
		if seen[v] {
			continue
		}
		seen[v] = true
		out = append(out, it)
	}
	return out, nil
}

// ShouldFilter 逐条降级路径（BatchFilter 不可用时）。
// 注意：逐条模式下无法跨 item 去重，始终返回 false。
// 去重语义仅在 FilterBatch 中保证。
func (f *DedupByFieldFilter) ShouldFilter(
	_ context.Context,
	_ *core.RecommendContext,
	_ *core.Item,
) (bool, error) {
	return false, nil
}
