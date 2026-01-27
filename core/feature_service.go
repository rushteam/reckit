package core

import "context"

// FeatureService 是特征服务的领域接口。
//
// 设计原则：
//   - 定义在领域层（core），由基础设施层（feature）实现
//   - 遵循依赖倒置原则：领域层定义接口，基础设施层实现接口
//   - 避免循环依赖：领域层不依赖基础设施层
//
// 使用场景：
//   - 获取用户特征：用户画像、历史行为等
//   - 获取物品特征：物品属性、统计特征等
//   - 获取实时特征：用户-物品交互特征、上下文特征等
//
// 实现：
//   - feature.BaseFeatureService 实现此接口
//   - 其他特征服务（Feast、Redis、HTTP 等）也可以实现此接口
type FeatureService interface {
	// Name 返回特征服务名称（用于日志/监控）
	Name() string

	// GetUserFeatures 获取用户特征（单个用户）
	GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error)

	// BatchGetUserFeatures 批量获取用户特征（推荐使用，减少网络往返）
	BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error)

	// GetItemFeatures 获取物品特征（单个物品）
	GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error)

	// BatchGetItemFeatures 批量获取物品特征（推荐使用，减少网络往返）
	BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error)

	// GetRealtimeFeatures 获取实时特征（用户-物品对）
	GetRealtimeFeatures(ctx context.Context, userID, itemID string) (map[string]float64, error)

	// BatchGetRealtimeFeatures 批量获取实时特征
	BatchGetRealtimeFeatures(ctx context.Context, pairs []FeatureUserItemPair) (map[FeatureUserItemPair]map[string]float64, error)

	// Close 关闭特征服务，释放资源
	Close() error
}

// FeatureUserItemPair 用户-物品对，用于实时特征查询
type FeatureUserItemPair struct {
	UserID string
	ItemID string
}
