package filter

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// ExposedFilter 是已曝光过滤器，过滤掉用户已经看过的物品。
// 支持两种数据源：
// 1. IDs 列表集合（近期数据）- 通过 GetExposedItems 获取
// 2. 布隆过滤器（较长周期数据，按天维度实现时间窗口）- 通过 CheckExposedInBloomFilter 检查
type ExposedFilter struct {
	// Store 用于从存储中读取用户曝光历史
	Store ExposedStore

	// KeyPrefix 是 Store 中的 key 前缀
	// 对于 IDs 列表：实际 key 为 {KeyPrefix}:{UserID}
	// 对于布隆过滤器：实际 key 为 {KeyPrefix}:bloom:{UserID}:{date}
	KeyPrefix string

	// TimeWindow 是曝光时间窗口（秒），用于 IDs 列表集合（近期数据）
	TimeWindow int64

	// BloomFilterDayWindow 是布隆过滤器的时间窗口（天数），用于较长周期数据
	// 如果为 0，则不使用布隆过滤器
	BloomFilterDayWindow int
}

// ExposedStore 是曝光历史存储接口。
type ExposedStore interface {
	// GetExposedItems 获取用户在指定时间窗口内已曝光的物品 ID 列表（近期数据）
	GetExposedItems(ctx context.Context, userID string, keyPrefix string, timeWindow int64) ([]string, error)

	// CheckExposedInBloomFilter 检查物品是否在布隆过滤器中（较长周期数据，按天维度）
	// dayWindow 是时间窗口（天数），检查最近 dayWindow 天内的布隆过滤器
	// 返回 true 表示可能在布隆过滤器中（存在误判可能），false 表示一定不在
	CheckExposedInBloomFilter(ctx context.Context, userID string, itemID string, keyPrefix string, dayWindow int) (bool, error)
}

// NewExposedFilter 创建一个已曝光过滤器。
// timeWindow 是 IDs 列表的时间窗口（秒），用于近期数据
// bloomFilterDayWindow 是布隆过滤器的时间窗口（天数），用于较长周期数据，如果为 0 则不使用布隆过滤器
func NewExposedFilter(storeAdapter *StoreAdapter, keyPrefix string, timeWindow int64, bloomFilterDayWindow int) *ExposedFilter {
	var store ExposedStore
	if storeAdapter != nil {
		store = storeAdapter
	}
	return &ExposedFilter{
		Store:                store,
		KeyPrefix:            keyPrefix,
		TimeWindow:           timeWindow,
		BloomFilterDayWindow: bloomFilterDayWindow,
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

	// 1. 检查 IDs 列表集合（近期数据）
	timeWindow := f.TimeWindow
	if timeWindow > 0 {
		exposedIDs, err := f.Store.GetExposedItems(ctx, rctx.UserID, keyPrefix, timeWindow)
		if err == nil {
			for _, id := range exposedIDs {
				if item.ID == id {
					return true, nil
				}
			}
		}
		// 如果 IDs 列表检查失败，继续检查布隆过滤器
	}

	// 2. 检查布隆过滤器（较长周期数据，按天维度）
	if f.BloomFilterDayWindow > 0 {
		exists, err := f.Store.CheckExposedInBloomFilter(ctx, rctx.UserID, item.ID, keyPrefix, f.BloomFilterDayWindow)
		if err == nil && exists {
			// 布隆过滤器返回 true 表示可能在过滤器中（存在误判可能）
			// 为了安全起见，我们将其视为已曝光
			return true, nil
		}
		// 如果布隆过滤器返回 false，表示一定不在，继续处理
	}

	return false, nil
}
