package filter

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// DiagnosticDetail 表示单个过滤器的判定细节。
type DiagnosticDetail struct {
	FilterName string
	Filtered   bool
	Err        error
}

// DiagnosticResult 是过滤诊断结果。
type DiagnosticResult struct {
	Filtered bool
	Details  []DiagnosticDetail
}

// DiagnoseItem 依次执行 filters 的 ShouldFilter，用于定位“某条 item 被谁过滤”。
// 注意：该函数用于诊断，默认按 ShouldFilter 执行，不走 BatchFilter。
func DiagnoseItem(
	ctx context.Context,
	rctx *core.RecommendContext,
	item *core.Item,
	filters []Filter,
) DiagnosticResult {
	res := DiagnosticResult{
		Details: make([]DiagnosticDetail, 0, len(filters)),
	}
	for _, f := range filters {
		if f == nil {
			continue
		}
		filtered, err := f.ShouldFilter(ctx, rctx, item)
		d := DiagnosticDetail{
			FilterName: f.Name(),
			Filtered:   filtered,
			Err:        err,
		}
		if filtered {
			res.Filtered = true
		}
		res.Details = append(res.Details, d)
	}
	return res
}
