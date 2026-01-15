package filter

import (
	"context"
	"encoding/json"
	"time"

	"github.com/rushteam/reckit/store"
)

// StoreAdapter 将 store.Store 适配为过滤器所需的存储接口。
type StoreAdapter struct {
	store store.Store
}

// NewStoreAdapter 创建一个 Store 适配器。
func NewStoreAdapter(s store.Store) *StoreAdapter {
	return &StoreAdapter{store: s}
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
