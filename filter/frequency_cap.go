package filter

import (
	"context"
	"time"

	"github.com/rushteam/reckit/core"
)

// FrequencyCapStore 提供 user-item 粒度的曝光计数查询。
type FrequencyCapStore interface {
	// GetImpressionCount 返回 user 对 item 在 window 内的曝光次数。
	GetImpressionCount(ctx context.Context, userID, itemID string, window time.Duration) (int, error)
}

// FrequencyCapFilter 曝光频次控制——同一物品在 Window 内对同一用户最多曝光 MaxCount 次。
type FrequencyCapFilter struct {
	Store    FrequencyCapStore
	MaxCount int
	Window   time.Duration
}

func (f *FrequencyCapFilter) Name() string { return "filter.frequency_cap" }

func (f *FrequencyCapFilter) ShouldFilter(
	ctx context.Context,
	rctx *core.RecommendContext,
	item *core.Item,
) (bool, error) {
	if item == nil {
		return true, nil
	}
	if f.Store == nil || f.MaxCount <= 0 || rctx == nil || rctx.UserID == "" {
		return false, nil
	}
	cnt, err := f.Store.GetImpressionCount(ctx, rctx.UserID, item.ID, f.Window)
	if err != nil {
		return false, nil
	}
	return cnt >= f.MaxCount, nil
}
