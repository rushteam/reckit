package filter

import (
	"context"
	"encoding/json"
	"strconv"
	"time"

	"reckit/store"
)

// StoreAdapter 将 store.Store 适配为过滤器所需的存储接口。
// 它同时实现了 BlacklistStore、UserBlockStore 和 ExposedStore 接口。
type StoreAdapter struct {
	store store.Store
}

// NewStoreAdapter 创建一个 Store 适配器。
func NewStoreAdapter(s store.Store) *StoreAdapter {
	return &StoreAdapter{store: s}
}

// GetBlacklist 从 Store 读取黑名单。
// 支持两种格式：
// 1. JSON 数组：["1", "2", "3"] 或 [1, 2, 3]
// 2. Set 格式（如果 Store 支持）
func (a *StoreAdapter) GetBlacklist(ctx context.Context, key string) ([]int64, error) {
	data, err := a.store.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	var ids []int64
	// 尝试解析为 JSON 数组
	if err := json.Unmarshal(data, &ids); err == nil {
		return ids, nil
	}

	// 尝试解析为字符串数组
	var strIDs []string
	if err := json.Unmarshal(data, &strIDs); err == nil {
		ids = make([]int64, 0, len(strIDs))
		for _, s := range strIDs {
			if id, err := strconv.ParseInt(s, 10, 64); err == nil {
				ids = append(ids, id)
			}
		}
		return ids, nil
	}

	return nil, err
}

// GetUserBlocks 从 Store 读取用户拉黑列表。
// key 格式：{keyPrefix}:{userID}
func (a *StoreAdapter) GetUserBlocks(ctx context.Context, userID int64, keyPrefix string) ([]int64, error) {
	key := keyPrefix + ":" + strconv.FormatInt(userID, 10)
	return a.GetBlacklist(ctx, key)
}

// GetExposedItems 从 Store 读取用户曝光历史。
// key 格式：{keyPrefix}:{userID}
// 支持两种格式：
// 1. 简单列表：只包含物品 ID
// 2. 带时间戳的列表：{"item_id": 123, "timestamp": 1234567890}
func (a *StoreAdapter) GetExposedItems(ctx context.Context, userID int64, keyPrefix string, timeWindow int64) ([]int64, error) {
	key := keyPrefix + ":" + strconv.FormatInt(userID, 10)
	data, err := a.store.Get(ctx, key)
	if err != nil {
		return nil, err
	}

	now := time.Now().Unix()
	cutoffTime := now - timeWindow

	// 尝试解析为简单 ID 列表
	var ids []int64
	if err := json.Unmarshal(data, &ids); err == nil {
		// 简单列表，全部返回
		return ids, nil
	}

	// 尝试解析为带时间戳的列表
	var items []struct {
		ItemID    int64 `json:"item_id"`
		Timestamp int64 `json:"timestamp"`
	}
	if err := json.Unmarshal(data, &items); err == nil {
		ids = make([]int64, 0, len(items))
		for _, item := range items {
			// 只返回时间窗口内的曝光
			if timeWindow > 0 && item.Timestamp < cutoffTime {
				continue
			}
			ids = append(ids, item.ItemID)
		}
		return ids, nil
	}

	// 尝试解析为字符串数组
	var strIDs []string
	if err := json.Unmarshal(data, &strIDs); err == nil {
		ids = make([]int64, 0, len(strIDs))
		for _, s := range strIDs {
			if id, err := strconv.ParseInt(s, 10, 64); err == nil {
				ids = append(ids, id)
			}
		}
		return ids, nil
	}

	return nil, err
}
