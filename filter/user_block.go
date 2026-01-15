package filter

import (
	"context"

	"reckit/core"
)

// UserBlockFilter 是用户拉黑过滤器，过滤掉用户拉黑的物品。
type UserBlockFilter struct {
	// Store 用于从存储中读取用户拉黑列表
	Store UserBlockStore

	// KeyPrefix 是 Store 中的 key 前缀，实际 key 为 {KeyPrefix}:{UserID}
	KeyPrefix string
}

// UserBlockStore 是用户拉黑存储接口。
type UserBlockStore interface {
	// GetUserBlocks 获取用户拉黑的物品 ID 列表
	GetUserBlocks(ctx context.Context, userID string, keyPrefix string) ([]string, error)
}

// NewUserBlockFilter 创建一个用户拉黑过滤器。
func NewUserBlockFilter(storeAdapter *StoreAdapter, keyPrefix string) *UserBlockFilter {
	var store UserBlockStore
	if storeAdapter != nil {
		store = storeAdapter
	}
	return &UserBlockFilter{
		Store:     store,
		KeyPrefix: keyPrefix,
	}
}

func (f *UserBlockFilter) Name() string {
	return "filter.user_block"
}

func (f *UserBlockFilter) ShouldFilter(
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
		keyPrefix = "user:block"
	}

	blockedIDs, err := f.Store.GetUserBlocks(ctx, rctx.UserID, keyPrefix)
	if err != nil {
		return false, nil
	}

	for _, id := range blockedIDs {
		if item.ID == id {
			return true, nil
		}
	}

	return false, nil
}
