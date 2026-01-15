package recall

import (
	"context"
	"encoding/json"
	"strconv"

	"reckit/store"
)

// MemoryCFStore 是基于内存的协同过滤存储实现（用于测试和示例）。
// 生产环境应该实现基于 Redis、MySQL 等的版本。
type MemoryCFStore struct {
	// userItems[userID][itemID] = score
	userItems map[int64]map[int64]float64

	// itemUsers[itemID][userID] = score
	itemUsers map[int64]map[int64]float64

	// 所有用户和物品的列表
	allUsers []int64
	allItems []int64
}

// NewMemoryCFStore 创建一个内存协同过滤存储。
func NewMemoryCFStore() *MemoryCFStore {
	return &MemoryCFStore{
		userItems: make(map[int64]map[int64]float64),
		itemUsers: make(map[int64]map[int64]float64),
		allUsers:  make([]int64, 0),
		allItems:  make([]int64, 0),
	}
}

// AddInteraction 添加用户-物品交互记录。
func (s *MemoryCFStore) AddInteraction(userID, itemID int64, score float64) {
	// 添加到 userItems
	if s.userItems[userID] == nil {
		s.userItems[userID] = make(map[int64]float64)
	}
	s.userItems[userID][itemID] = score

	// 添加到 itemUsers
	if s.itemUsers[itemID] == nil {
		s.itemUsers[itemID] = make(map[int64]float64)
	}
	s.itemUsers[itemID][userID] = score

	// 更新用户列表
	userExists := false
	for _, uid := range s.allUsers {
		if uid == userID {
			userExists = true
			break
		}
	}
	if !userExists {
		s.allUsers = append(s.allUsers, userID)
	}

	// 更新物品列表
	itemExists := false
	for _, iid := range s.allItems {
		if iid == itemID {
			itemExists = true
			break
		}
	}
	if !itemExists {
		s.allItems = append(s.allItems, itemID)
	}
}

func (s *MemoryCFStore) GetUserItems(ctx context.Context, userID int64) (map[int64]float64, error) {
	if items, ok := s.userItems[userID]; ok {
		// 返回副本
		result := make(map[int64]float64, len(items))
		for k, v := range items {
			result[k] = v
		}
		return result, nil
	}
	return make(map[int64]float64), nil
}

func (s *MemoryCFStore) GetItemUsers(ctx context.Context, itemID int64) (map[int64]float64, error) {
	if users, ok := s.itemUsers[itemID]; ok {
		// 返回副本
		result := make(map[int64]float64, len(users))
		for k, v := range users {
			result[k] = v
		}
		return result, nil
	}
	return make(map[int64]float64), nil
}

func (s *MemoryCFStore) GetAllUsers(ctx context.Context) ([]int64, error) {
	result := make([]int64, len(s.allUsers))
	copy(result, s.allUsers)
	return result, nil
}

func (s *MemoryCFStore) GetAllItems(ctx context.Context) ([]int64, error) {
	result := make([]int64, len(s.allItems))
	copy(result, s.allItems)
	return result, nil
}

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

func (a *StoreCFAdapter) GetUserItems(ctx context.Context, userID int64) (map[int64]float64, error) {
	key := a.KeyPrefix + ":user:" + strconv.FormatInt(userID, 10)
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
			return make(map[int64]float64), nil
		}
		return nil, err
	}

	var result map[int64]float64
	// 尝试解析 JSON
	if err := json.Unmarshal(data, &result); err != nil {
		// 如果失败，尝试解析为 map[string]float64 然后转换
		var strMap map[string]float64
		if err2 := json.Unmarshal(data, &strMap); err2 != nil {
			return nil, err
		}
		result = make(map[int64]float64, len(strMap))
		for k, v := range strMap {
			if id, err := strconv.ParseInt(k, 10, 64); err == nil {
				result[id] = v
			}
		}
	}

	return result, nil
}

func (a *StoreCFAdapter) GetItemUsers(ctx context.Context, itemID int64) (map[int64]float64, error) {
	key := a.KeyPrefix + ":item:" + strconv.FormatInt(itemID, 10)
	data, err := a.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
			return make(map[int64]float64), nil
		}
		return nil, err
	}

	var result map[int64]float64
	// 尝试解析 JSON
	if err := json.Unmarshal(data, &result); err != nil {
		// 如果失败，尝试解析为 map[string]float64 然后转换
		var strMap map[string]float64
		if err2 := json.Unmarshal(data, &strMap); err2 != nil {
			return nil, err
		}
		result = make(map[int64]float64, len(strMap))
		for k, v := range strMap {
			if id, err := strconv.ParseInt(k, 10, 64); err == nil {
				result[id] = v
			}
		}
	}

	return result, nil
}

func (a *StoreCFAdapter) GetAllUsers(ctx context.Context) ([]int64, error) {
	key := a.KeyPrefix + ":users"
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

func (a *StoreCFAdapter) GetAllItems(ctx context.Context) ([]int64, error) {
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
