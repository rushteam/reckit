package filter

import (
	"context"
	"time"

	"github.com/rushteam/reckit/core"
)

// TimeDecayFilter 按时间过滤过期内容。
// 从 item.Meta[TimeField] 读取时间戳（支持 time.Time / int64 秒 / float64 秒），
// 距今超过 MaxAge 的物品被过滤。
type TimeDecayFilter struct {
	TimeField string
	MaxAge    time.Duration
	// NowFunc 可选：返回当前时间，便于测试。
	NowFunc func() time.Time
}

func (f *TimeDecayFilter) Name() string { return "filter.time_decay" }

func (f *TimeDecayFilter) ShouldFilter(
	_ context.Context,
	_ *core.RecommendContext,
	item *core.Item,
) (bool, error) {
	if item == nil {
		return true, nil
	}
	if f.TimeField == "" || f.MaxAge <= 0 {
		return false, nil
	}

	raw, ok := item.Meta[f.TimeField]
	if !ok {
		return false, nil
	}

	var ts time.Time
	switch v := raw.(type) {
	case time.Time:
		ts = v
	case int64:
		ts = time.Unix(v, 0)
	case float64:
		ts = time.Unix(int64(v), 0)
	default:
		return false, nil
	}

	now := time.Now()
	if f.NowFunc != nil {
		now = f.NowFunc()
	}
	return now.Sub(ts) > f.MaxAge, nil
}
