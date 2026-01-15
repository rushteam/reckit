package filter

import (
	"context"

	"reckit/core"
)

// ExposedFilter 是已曝光过滤器，过滤掉用户已经看过的物品。
type ExposedFilter struct {
	// Store 用于从存储中读取用户曝光历史
	Store ExposedStore

	// KeyPrefix 是 Store 中的 key 前缀，实际 key 为 {KeyPrefix}:{UserID}
	KeyPrefix string

	// TimeWindow 是曝光时间窗口（秒）
	TimeWindow int64
}

// ExposedStore 是曝光历史存储接口。
type ExposedStore interface {
	// GetExposedItems 获取用户在指定时间窗口内已曝光的物品 ID 列表
	GetExposedItems(ctx context.Context, userID string, keyPrefix string, timeWindow int64) ([]string, error)
}

// NewExposedFilter 创建一个已曝光过滤器。
func NewExposedFilter(storeAdapter *StoreAdapter, keyPrefix string, timeWindow int64) *ExposedFilter {
	var store ExposedStore
	if storeAdapter != nil {
		store = storeAdapter
	}
	return &ExposedFilter{
		Store:      store,
		KeyPrefix:  keyPrefix,
		TimeWindow: timeWindow,
	}
}

func (f *ExposedFilter) Name() string {
	return "filter.exposed"
}

func (f *ExposedFilter) ShouldFilter(
	ctx context.Context,
	rctx *core.RecommendContext,
	item *core.Item,
) (bool, error) {
	if item == nil || rctx == nil || rctx.UserID == "" {
		return false, nil
	}

	if f.Store == nil {
		return false, nil
	}

	keyPrefix := f.KeyPrefix
	if keyPrefix == "" {
		keyPrefix = "user:exposed"
	}

	timeWindow := f.TimeWindow
	if timeWindow == 0 {
		timeWindow = 7 * 24 * 3600
	}

	exposedIDs, err := f.Store.GetExposedItems(ctx, rctx.UserID, keyPrefix, timeWindow)
	if err != nil {
		return false, nil
	}

	for _, id := range exposedIDs {
		if item.ID == id {
			return true, nil
		}
	}

	return false, nil
}
