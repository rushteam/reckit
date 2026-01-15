package recall

import (
	"context"
	"encoding/json"

	"github.com/rushteam/reckit/store"
)

// StoreCFAdapter 是基于 Store 接口的协同过滤存储适配器。
// 从 Redis/MySQL 等存储中读取用户-物品交互数据。
type StoreCFAdapter struct {
	store store.Store

	// KeyPrefix 是存储 key 的前缀
	// 用户物品交互：{KeyPrefix}:user:{userID}
	// 物品用户交互：{KeyPrefix}:item:{itemID}
	// 所有用户列表：{KeyPrefix}:users
	// 所有物品列表：{KeyPrefix}:items
	KeyPrefix string
}

// NewStoreCFAdapter 创建一个基于 Store 的协同过滤适配器。
func NewStoreCFAdapter(s store.Store, keyPrefix string) *StoreCFAdapter {
	if keyPrefix == "" {
		keyPrefix = "cf"
	}
	return &StoreCFAdapter{
		store:     s,
		KeyPrefix: keyPrefix,
	}
}

func (a *StoreCFAdapter) GetUserItems(ctx context.Context, userID string) (map[string]float64, error) {
	key := a.KeyPrefix + ":user:" + userID
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
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

func (a *StoreCFAdapter) GetItemUsers(ctx context.Context, itemID string) (map[string]float64, error) {
	key := a.KeyPrefix + ":item:" + itemID
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
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

func (a *StoreCFAdapter) GetAllUsers(ctx context.Context) ([]string, error) {
	key := a.KeyPrefix + ":users"
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
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

func (a *StoreCFAdapter) GetAllItems(ctx context.Context) ([]string, error) {
	key := a.KeyPrefix + ":items"
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
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

// SetupCFTestData 辅助函数：为测试准备协同过滤数据到 Store 中。
// 使用 StoreCFAdapter + MemoryStore 时，可以用这个函数方便地添加测试数据。
func SetupCFTestData(ctx context.Context, adapter *StoreCFAdapter, interactions []struct {
	UserID string
	ItemID string
	Score  float64
}) error {
	// 收集所有用户和物品
	userItems := make(map[string]map[string]float64)
	itemUsers := make(map[string]map[string]float64)
	allUsers := make(map[string]bool)
	allItems := make(map[string]bool)

	for _, inter := range interactions {
		// 添加到 userItems
		if userItems[inter.UserID] == nil {
			userItems[inter.UserID] = make(map[string]float64)
		}
		userItems[inter.UserID][inter.ItemID] = inter.Score

		// 添加到 itemUsers
		if itemUsers[inter.ItemID] == nil {
			itemUsers[inter.ItemID] = make(map[string]float64)
		}
		itemUsers[inter.ItemID][inter.UserID] = inter.Score

		allUsers[inter.UserID] = true
		allItems[inter.ItemID] = true
	}

	// 写入用户物品交互数据
	for userID, items := range userItems {
		key := adapter.KeyPrefix + ":user:" + userID
		data, err := json.Marshal(items)
		if err != nil {
			return err
		}
		if err := adapter.store.Set(ctx, key, data); err != nil {
			return err
		}
	}

	// 写入物品用户交互数据
	for itemID, users := range itemUsers {
		key := adapter.KeyPrefix + ":item:" + itemID
		data, err := json.Marshal(users)
		if err != nil {
			return err
		}
		if err := adapter.store.Set(ctx, key, data); err != nil {
			return err
		}
	}

	// 写入所有用户列表
	userList := make([]string, 0, len(allUsers))
	for userID := range allUsers {
		userList = append(userList, userID)
	}
	usersKey := adapter.KeyPrefix + ":users"
	usersData, err := json.Marshal(userList)
	if err != nil {
		return err
	}
	if err := adapter.store.Set(ctx, usersKey, usersData); err != nil {
		return err
	}

	// 写入所有物品列表
	itemList := make([]string, 0, len(allItems))
	for itemID := range allItems {
		itemList = append(itemList, itemID)
	}
	itemsKey := adapter.KeyPrefix + ":items"
	itemsData, err := json.Marshal(itemList)
	if err != nil {
		return err
	}
	return adapter.store.Set(ctx, itemsKey, itemsData)
}
