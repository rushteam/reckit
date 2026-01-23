package recall

import (
	"context"
	"encoding/json"

	"github.com/rushteam/reckit/core"
)

// StoreContentAdapter 是基于 core.Store 接口的内容推荐存储适配器。
// 从 Redis/MySQL 等存储中读取物品特征和用户偏好。
type StoreContentAdapter struct {
	store core.Store

	// KeyPrefix 是存储 key 的前缀
	// 物品特征：{KeyPrefix}:item:{itemID}
	// 用户偏好：{KeyPrefix}:user:{userID}
	// 所有物品列表：{KeyPrefix}:items
	KeyPrefix string
}

// NewStoreContentAdapter 创建一个基于 core.Store 的内容推荐适配器。
func NewStoreContentAdapter(s core.Store, keyPrefix string) *StoreContentAdapter {
	if keyPrefix == "" {
		keyPrefix = "content"
	}
	return &StoreContentAdapter{
		store:     s,
		KeyPrefix: keyPrefix,
	}
}

func (a *StoreContentAdapter) GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error) {
	key := a.KeyPrefix + ":item:" + itemID
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return make(map[string]float64), nil
		}
		return nil, err
	}

	var result map[string]float64
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}

	return result, nil
}

func (a *StoreContentAdapter) GetUserPreferences(ctx context.Context, userID string) (map[string]float64, error) {
	key := a.KeyPrefix + ":user:" + userID
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return make(map[string]float64), nil
		}
		return nil, err
	}

	var result map[string]float64
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}

	return result, nil
}

func (a *StoreContentAdapter) GetSimilarItems(ctx context.Context, itemFeatures map[string]float64, topK int) ([]string, error) {
	return []string{}, nil
}

func (a *StoreContentAdapter) GetAllItems(ctx context.Context) ([]string, error) {
	key := a.KeyPrefix + ":items"
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return []string{}, nil
		}
		return nil, err
	}

	var result []string
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// 确保实现 ContentStore 接口
var _ ContentStore = (*StoreContentAdapter)(nil)
