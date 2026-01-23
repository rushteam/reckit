package recall

import (
	"context"
	"encoding/json"

	"github.com/rushteam/reckit/core"
)

// StoreMFAdapter 是基于 core.Store 接口的矩阵分解存储适配器。
// 从 Redis/MySQL 等存储中读取用户和物品的隐向量。
type StoreMFAdapter struct {
	store core.Store

	// KeyPrefix 是存储 key 的前缀
	// 用户隐向量：{KeyPrefix}:user:{userID}
	// 物品隐向量：{KeyPrefix}:item:{itemID}
	// 所有物品列表：{KeyPrefix}:items
	KeyPrefix string
}

// NewStoreMFAdapter 创建一个基于 core.Store 的矩阵分解适配器。
func NewStoreMFAdapter(s core.Store, keyPrefix string) *StoreMFAdapter {
	if keyPrefix == "" {
		keyPrefix = "mf"
	}
	return &StoreMFAdapter{
		store:     s,
		KeyPrefix: keyPrefix,
	}
}

func (a *StoreMFAdapter) GetUserVector(ctx context.Context, userID string) ([]float64, error) {
	key := a.KeyPrefix + ":user:" + userID
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return []float64{}, nil
		}
		return nil, err
	}

	var result []float64
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (a *StoreMFAdapter) GetItemVector(ctx context.Context, itemID string) ([]float64, error) {
	key := a.KeyPrefix + ":item:" + itemID
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return []float64{}, nil
		}
		return nil, err
	}

	var result []float64
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return result, nil
}

func (a *StoreMFAdapter) GetAllItemVectors(ctx context.Context) (map[string][]float64, error) {
	// 先获取所有物品列表
	itemsKey := a.KeyPrefix + ":items"
	itemsData, err := a.store.Get(ctx, itemsKey)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return make(map[string][]float64), nil
		}
		return nil, err
	}

	var itemIDs []string
	if err := json.Unmarshal(itemsData, &itemIDs); err != nil {
		return nil, err
	}

	// 批量获取物品向量
	result := make(map[string][]float64)
	for _, itemID := range itemIDs {
		vector, err := a.GetItemVector(ctx, itemID)
		if err != nil {
			continue
		}
		if len(vector) > 0 {
			result[itemID] = vector
		}
	}

	return result, nil
}

// 确保实现 MFStore 接口
var _ MFStore = (*StoreMFAdapter)(nil)
