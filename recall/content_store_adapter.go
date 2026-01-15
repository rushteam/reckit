package recall

import (
	"context"
	"encoding/json"
	"strconv"

	"reckit/store"
)

// StoreContentAdapter 是基于 Store 接口的内容推荐存储适配器。
// 从 Redis/MySQL 等存储中读取物品特征和用户偏好。
type StoreContentAdapter struct {
	store store.Store

	// KeyPrefix 是存储 key 的前缀
	// 物品特征：{KeyPrefix}:item:{itemID}
	// 用户偏好：{KeyPrefix}:user:{userID}
	// 所有物品列表：{KeyPrefix}:items
	KeyPrefix string
}

// NewStoreContentAdapter 创建一个基于 Store 的内容推荐适配器。
func NewStoreContentAdapter(s store.Store, keyPrefix string) *StoreContentAdapter {
	if keyPrefix == "" {
		keyPrefix = "content"
	}
	return &StoreContentAdapter{
		store:     s,
		KeyPrefix: keyPrefix,
	}
}

func (a *StoreContentAdapter) GetItemFeatures(ctx context.Context, itemID int64) (map[string]float64, error) {
	key := a.KeyPrefix + ":item:" + strconv.FormatInt(itemID, 10)
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
			return make(map[string]float64), nil
		}
		return nil, err
	}

	var result map[string]float64
	if err := json.Unmarshal(data, &result); err != nil {
		// 尝试解析为 map[string]interface{} 然后转换
		var strMap map[string]interface{}
		if err2 := json.Unmarshal(data, &strMap); err2 != nil {
			return nil, err
		}
		result = make(map[string]float64, len(strMap))
		for k, v := range strMap {
			if fv, ok := v.(float64); ok {
				result[k] = fv
			}
		}
	}

	return result, nil
}

func (a *StoreContentAdapter) GetUserPreferences(ctx context.Context, userID int64) (map[string]float64, error) {
	key := a.KeyPrefix + ":user:" + strconv.FormatInt(userID, 10)
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
			return make(map[string]float64), nil
		}
		return nil, err
	}

	var result map[string]float64
	if err := json.Unmarshal(data, &result); err != nil {
		// 尝试解析为 map[string]interface{} 然后转换
		var strMap map[string]interface{}
		if err2 := json.Unmarshal(data, &strMap); err2 != nil {
			return nil, err
		}
		result = make(map[string]float64, len(strMap))
		for k, v := range strMap {
			if fv, ok := v.(float64); ok {
				result[k] = fv
			}
		}
	}

	return result, nil
}

func (a *StoreContentAdapter) GetSimilarItems(ctx context.Context, itemFeatures map[string]float64, topK int) ([]int64, error) {
	// 简化实现：返回空，表示不支持优化
	// 生产环境可以实现基于特征索引的快速检索
	return []int64{}, nil
}

func (a *StoreContentAdapter) GetAllItems(ctx context.Context) ([]int64, error) {
	key := a.KeyPrefix + ":items"
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
			return []int64{}, nil
		}
		return nil, err
	}

	var result []int64
	if err := json.Unmarshal(data, &result); err != nil {
		return nil, err
	}
	return result, nil
}

// 确保实现 ContentStore 接口
var _ ContentStore = (*StoreContentAdapter)(nil)
