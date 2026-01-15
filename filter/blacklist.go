package filter

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// BlacklistFilter 是黑名单过滤器，过滤掉黑名单中的物品。
type BlacklistFilter struct {
	// ItemIDs 是内存中的黑名单物品 ID 列表
	ItemIDs []string

	// Store 用于从存储中读取黑名单（可选）
	Store BlacklistStore

	// Key 是 Store 中的黑名单 key（可选）
	Key string
}

// BlacklistStore 是黑名单存储接口。
type BlacklistStore interface {
	// GetBlacklist 获取黑名单物品 ID 列表
	GetBlacklist(ctx context.Context, key string) ([]string, error)
}

// NewBlacklistFilter 创建一个黑名单过滤器。
func NewBlacklistFilter(itemIDs []string, storeAdapter *StoreAdapter, key string) *BlacklistFilter {
	var store BlacklistStore
	if storeAdapter != nil {
		store = storeAdapter
	}
	return &BlacklistFilter{
		ItemIDs: itemIDs,
		Store:   store,
		Key:     key,
	}
}

func (f *BlacklistFilter) Name() string {
	return "filter.blacklist"
}

func (f *BlacklistFilter) ShouldFilter(
	ctx context.Context,
	_ *core.RecommendContext,
	item *core.Item,
) (bool, error) {
	if item == nil {
		return true, nil
	}

	// 从内存列表检查
	for _, id := range f.ItemIDs {
		if item.ID == id {
			return true, nil
		}
	}

	// 从 Store 检查
	if f.Store != nil && f.Key != "" {
		blacklist, err := f.Store.GetBlacklist(ctx, f.Key)
		if err == nil {
			for _, id := range blacklist {
				if item.ID == id {
					return true, nil
				}
			}
		}
	}

	return false, nil
}
