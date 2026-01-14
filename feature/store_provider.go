package feature

import (
	"context"
	"encoding/json"
	"fmt"

	"reckit/store"
)

// StoreFeatureProvider 是基于 Store 的特征提供者实现，采用适配器模式。
// 将 store.Store 适配为 FeatureProvider 接口。
type StoreFeatureProvider struct {
	store      store.Store
	keyPrefix  KeyPrefix
	serializer FeatureSerializer
}

// KeyPrefix 定义特征存储的 key 前缀
type KeyPrefix struct {
	User     string // 用户特征前缀，例如 "user:features:"
	Item     string // 物品特征前缀，例如 "item:features:"
	Realtime string // 实时特征前缀，例如 "realtime:features:"
}

// FeatureSerializer 是特征序列化接口，支持不同的序列化格式（JSON、MsgPack等）
type FeatureSerializer interface {
	Serialize(features map[string]float64) ([]byte, error)
	Deserialize(data []byte) (map[string]float64, error)
}

// JSONSerializer 是 JSON 序列化实现
type JSONSerializer struct{}

func (j *JSONSerializer) Serialize(features map[string]float64) ([]byte, error) {
	return json.Marshal(features)
}

func (j *JSONSerializer) Deserialize(data []byte) (map[string]float64, error) {
	var features map[string]float64
	if err := json.Unmarshal(data, &features); err != nil {
		return nil, err
	}
	return features, nil
}

// NewStoreFeatureProvider 创建基于 Store 的特征提供者
func NewStoreFeatureProvider(store store.Store, keyPrefix KeyPrefix) *StoreFeatureProvider {
	if keyPrefix.User == "" {
		keyPrefix.User = "user:features:"
	}
	if keyPrefix.Item == "" {
		keyPrefix.Item = "item:features:"
	}
	if keyPrefix.Realtime == "" {
		keyPrefix.Realtime = "realtime:features:"
	}

	return &StoreFeatureProvider{
		store:      store,
		keyPrefix:  keyPrefix,
		serializer: &JSONSerializer{},
	}
}

// WithSerializer 设置序列化器
func (p *StoreFeatureProvider) WithSerializer(serializer FeatureSerializer) *StoreFeatureProvider {
	p.serializer = serializer
	return p
}

func (p *StoreFeatureProvider) Name() string {
	return fmt.Sprintf("store.%s", p.store.Name())
}

func (p *StoreFeatureProvider) GetUserFeatures(ctx context.Context, userID int64) (map[string]float64, error) {
	key := fmt.Sprintf("%s%d", p.keyPrefix.User, userID)
	data, err := p.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
			return nil, ErrFeatureNotFound
		}
		return nil, err
	}

	return p.serializer.Deserialize(data)
}

func (p *StoreFeatureProvider) BatchGetUserFeatures(ctx context.Context, userIDs []int64) (map[int64]map[string]float64, error) {
	if len(userIDs) == 0 {
		return make(map[int64]map[string]float64), nil
	}

	// 构建 keys
	keys := make([]string, len(userIDs))
	keyToUserID := make(map[string]int64, len(userIDs))
	for i, userID := range userIDs {
		key := fmt.Sprintf("%s%d", p.keyPrefix.User, userID)
		keys[i] = key
		keyToUserID[key] = userID
	}

	// 批量获取
	dataMap, err := p.store.BatchGet(ctx, keys)
	if err != nil {
		return nil, err
	}

	// 反序列化
	result := make(map[int64]map[string]float64)
	for key, data := range dataMap {
		userID := keyToUserID[key]
		features, err := p.serializer.Deserialize(data)
		if err != nil {
			continue // 跳过反序列化失败的特征
		}
		result[userID] = features
	}

	return result, nil
}

func (p *StoreFeatureProvider) GetItemFeatures(ctx context.Context, itemID int64) (map[string]float64, error) {
	key := fmt.Sprintf("%s%d", p.keyPrefix.Item, itemID)
	data, err := p.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
			return nil, ErrFeatureNotFound
		}
		return nil, err
	}

	return p.serializer.Deserialize(data)
}

func (p *StoreFeatureProvider) BatchGetItemFeatures(ctx context.Context, itemIDs []int64) (map[int64]map[string]float64, error) {
	if len(itemIDs) == 0 {
		return make(map[int64]map[string]float64), nil
	}

	// 构建 keys
	keys := make([]string, len(itemIDs))
	keyToItemID := make(map[string]int64, len(itemIDs))
	for i, itemID := range itemIDs {
		key := fmt.Sprintf("%s%d", p.keyPrefix.Item, itemID)
		keys[i] = key
		keyToItemID[key] = itemID
	}

	// 批量获取
	dataMap, err := p.store.BatchGet(ctx, keys)
	if err != nil {
		return nil, err
	}

	// 反序列化
	result := make(map[int64]map[string]float64)
	for key, data := range dataMap {
		itemID := keyToItemID[key]
		features, err := p.serializer.Deserialize(data)
		if err != nil {
			continue // 跳过反序列化失败的特征
		}
		result[itemID] = features
	}

	return result, nil
}

func (p *StoreFeatureProvider) GetRealtimeFeatures(ctx context.Context, userID, itemID int64) (map[string]float64, error) {
	key := fmt.Sprintf("%s%d:%d", p.keyPrefix.Realtime, userID, itemID)
	data, err := p.store.Get(ctx, key)
	if err != nil {
		if err == store.ErrNotFound {
			return nil, ErrFeatureNotFound
		}
		return nil, err
	}

	return p.serializer.Deserialize(data)
}

func (p *StoreFeatureProvider) BatchGetRealtimeFeatures(ctx context.Context, pairs []UserItemPair) (map[UserItemPair]map[string]float64, error) {
	if len(pairs) == 0 {
		return make(map[UserItemPair]map[string]float64), nil
	}

	// 构建 keys
	keys := make([]string, len(pairs))
	pairToKey := make(map[string]UserItemPair, len(pairs))
	for i, pair := range pairs {
		key := fmt.Sprintf("%s%d:%d", p.keyPrefix.Realtime, pair.UserID, pair.ItemID)
		keys[i] = key
		pairToKey[key] = pair
	}

	// 批量获取
	dataMap, err := p.store.BatchGet(ctx, keys)
	if err != nil {
		return nil, err
	}

	// 反序列化
	result := make(map[UserItemPair]map[string]float64)
	for key, data := range dataMap {
		pair := pairToKey[key]
		features, err := p.serializer.Deserialize(data)
		if err != nil {
			continue // 跳过反序列化失败的特征
		}
		result[pair] = features
	}

	return result, nil
}
