package feature

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/rushteam/reckit/core"
)

// StoreFeatureProvider 是基于 core.Store 的特征提供者实现，采用适配器模式。
// 将 core.Store 适配为 FeatureProvider 接口。
type StoreFeatureProvider struct {
	store      core.Store
	keyPrefix  KeyPrefix
	serializer FeatureSerializer
}

// KeyPrefix 定义特征存储的 key 前缀
type KeyPrefix struct {
	User string // 用户特征前缀，例如 "user:features:"
	Item string // 物品特征前缀，例如 "item:features:"
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

// NewStoreFeatureProvider 创建基于 core.Store 的特征提供者
func NewStoreFeatureProvider(store core.Store, keyPrefix KeyPrefix) *StoreFeatureProvider {
	if keyPrefix.User == "" {
		keyPrefix.User = "user:features:"
	}
	if keyPrefix.Item == "" {
		keyPrefix.Item = "item:features:"
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

func (p *StoreFeatureProvider) GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error) {
	key := fmt.Sprintf("%s%s", p.keyPrefix.User, userID)
	data, err := p.store.Get(ctx, key)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return nil, ErrFeatureNotFound
		}
		return nil, err
	}

	return p.serializer.Deserialize(data)
}

func (p *StoreFeatureProvider) BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error) {
	if len(userIDs) == 0 {
		return make(map[string]map[string]float64), nil
	}

	// 构建 keys
	keys := make([]string, len(userIDs))
	keyToUserID := make(map[string]string, len(userIDs))
	for i, userID := range userIDs {
		key := fmt.Sprintf("%s%s", p.keyPrefix.User, userID)
		keys[i] = key
		keyToUserID[key] = userID
	}

	// 批量获取
	dataMap, err := p.store.BatchGet(ctx, keys)
	if err != nil {
		return nil, err
	}

	// 反序列化
	result := make(map[string]map[string]float64)
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

func (p *StoreFeatureProvider) GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error) {
	key := fmt.Sprintf("%s%s", p.keyPrefix.Item, itemID)
	data, err := p.store.Get(ctx, key)
	if err != nil {
		if core.IsStoreNotFound(err) {
			return nil, ErrFeatureNotFound
		}
		return nil, err
	}

	return p.serializer.Deserialize(data)
}

func (p *StoreFeatureProvider) BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error) {
	if len(itemIDs) == 0 {
		return make(map[string]map[string]float64), nil
	}

	// 构建 keys
	keys := make([]string, len(itemIDs))
	keyToItemID := make(map[string]string, len(itemIDs))
	for i, itemID := range itemIDs {
		key := fmt.Sprintf("%s%s", p.keyPrefix.Item, itemID)
		keys[i] = key
		keyToItemID[key] = itemID
	}

	// 批量获取
	dataMap, err := p.store.BatchGet(ctx, keys)
	if err != nil {
		return nil, err
	}

	// 反序列化
	result := make(map[string]map[string]float64)
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

