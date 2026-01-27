package filter

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
)

// BloomFilterChecker 是布隆过滤器检查器接口。
// 用户可以通过实现此接口来提供自定义的布隆过滤器检查逻辑。
type BloomFilterChecker interface {
	// CheckInBloomFilter 检查 itemID 是否在指定日期的布隆过滤器中
	// key 是布隆过滤器的存储 key，格式为 {keyPrefix}:bloom:{userID}:{date}
	// 返回 true 表示可能在布隆过滤器中（存在误判可能），false 表示一定不在
	CheckInBloomFilter(ctx context.Context, key string, itemID string) (bool, error)
}

// StoreAdapter 将 core.Store 适配为过滤器所需的存储接口。
type StoreAdapter struct {
	store core.Store

	// BloomFilterChecker 是可选的布隆过滤器检查器
	// 如果为 nil，CheckExposedInBloomFilter 将返回 false（未实现）
	BloomFilterChecker BloomFilterChecker
}

// NewStoreAdapter 创建一个 core.Store 适配器。
func NewStoreAdapter(s core.Store) *StoreAdapter {
	return &StoreAdapter{store: s}
}

// NewStoreAdapterWithBloomFilter 创建一个带布隆过滤器检查器的 core.Store 适配器。
func NewStoreAdapterWithBloomFilter(s core.Store, checker BloomFilterChecker) *StoreAdapter {
	return &StoreAdapter{
		store:              s,
		BloomFilterChecker: checker,
	}
}

// GetBlacklist 从 Store 读取黑名单。
func (a *StoreAdapter) GetBlacklist(ctx context.Context, key string) ([]string, error) {
	data, err := a.store.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var ids []string
	if err := json.Unmarshal(data, &ids); err != nil {
		return nil, err
	}

	return ids, nil
}

// GetUserBlocks 从 Store 读取用户拉黑列表。
func (a *StoreAdapter) GetUserBlocks(ctx context.Context, userID string, keyPrefix string) ([]string, error) {
	key := keyPrefix + ":" + userID
	return a.GetBlacklist(ctx, key)
}

// GetExposedItems 从 Store 读取用户曝光历史。
func (a *StoreAdapter) GetExposedItems(ctx context.Context, userID string, keyPrefix string, timeWindow int64) ([]string, error) {
	key := keyPrefix + ":" + userID
	data, err := a.store.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	now := time.Now().Unix()
	cutoffTime := now - timeWindow

	// 尝试解析为简单 ID 列表
	var ids []string
	if err := json.Unmarshal(data, &ids); err == nil {
		return ids, nil
	}

	// 尝试解析为带时间戳的列表
	var items []struct {
		ItemID    string `json:"item_id"`
		Timestamp int64  `json:"timestamp"`
	}
	if err := json.Unmarshal(data, &items); err == nil {
		ids = make([]string, 0, len(items))
		for _, item := range items {
			if timeWindow > 0 && item.Timestamp < cutoffTime {
				continue
			}
			ids = append(ids, item.ItemID)
		}
		return ids, nil
	}

	return nil, err
}

// CheckExposedInBloomFilter 检查物品是否在布隆过滤器中（较长周期数据，按天维度）。
// dayWindow 是时间窗口（天数），检查最近 dayWindow 天内的布隆过滤器。
// 返回 true 表示可能在布隆过滤器中（存在误判可能），false 表示一定不在。
//
// 布隆过滤器的 key 格式：{keyPrefix}:bloom:{userID}:{date}，其中 date 为 YYYYMMDD 格式。
//
// 注意：此方法需要设置 BloomFilterChecker，否则返回 false（未实现）。
// 用户可以通过扩展包实现具体的布隆过滤器检查逻辑，例如基于 Redis 的布隆过滤器。
func (a *StoreAdapter) CheckExposedInBloomFilter(ctx context.Context, userID string, itemID string, keyPrefix string, dayWindow int) (bool, error) {
	if a.BloomFilterChecker == nil {
		// 未实现布隆过滤器检查器，返回 false（表示未实现）
		return false, nil
	}

	if dayWindow <= 0 {
		return false, nil
	}

	// 检查最近 dayWindow 天内的布隆过滤器
	now := time.Now()
	for i := 0; i < dayWindow; i++ {
		date := now.AddDate(0, 0, -i)
		dateStr := date.Format("20060102") // YYYYMMDD 格式
		key := fmt.Sprintf("%s:bloom:%s:%s", keyPrefix, userID, dateStr)

		exists, err := a.BloomFilterChecker.CheckInBloomFilter(ctx, key, itemID)
		if err != nil {
			// 如果某个日期的布隆过滤器检查失败，继续检查其他日期
			continue
		}
		if exists {
			// 在任意一天的布隆过滤器中找到，返回 true
			return true, nil
		}
	}

	// 所有日期的布隆过滤器都返回 false，表示一定不在
	return false, nil
}
