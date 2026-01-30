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
//
// 注意：请求级上下文特征（如 latitude、time_of_day 等）应通过 RecommendContext.Params 传递，
// 而不是通过 FeatureService 获取。
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

	// Close 关闭特征服务，释放资源
	Close(ctx context.Context) error
}
