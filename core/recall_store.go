package core

import "context"

// RecallDataStore 是召回数据存储的领域接口。
//
// 设计原则：
//   - 定义在领域层（core），由基础设施层（recall）实现
//   - 遵循依赖倒置原则：领域层定义接口，基础设施层实现接口
//   - 统一召回算法的数据访问接口，避免接口爆炸
//
// 使用场景：
//   - 协同过滤：用户-物品交互数据
//   - 内容推荐：物品特征、用户偏好
//   - 矩阵分解：用户隐向量、物品隐向量
//   - 其他召回算法：统一的数据访问接口
//
// 实现：
//   - recall.StoreCFAdapter 实现此接口（基于 core.Store）
//   - recall.StoreContentAdapter 实现此接口（基于 core.Store）
//   - recall.StoreMFAdapter 实现此接口（基于 core.Store）
//   - 其他存储后端也可以实现此接口
type RecallDataStore interface {
	// Name 返回存储后端名称（用于日志/监控）
	Name() string

	// ========== 协同过滤数据 ==========

	// GetUserItems 获取用户交互过的物品及其评分/权重
	// 返回 map[itemID]score，score 可以是评分、点击次数、时长等
	GetUserItems(ctx context.Context, userID string) (map[string]float64, error)

	// GetItemUsers 获取与物品交互过的用户及其评分/权重
	// 返回 map[userID]score
	GetItemUsers(ctx context.Context, itemID string) (map[string]float64, error)

	// ========== 内容推荐数据 ==========

	// GetItemFeatures 获取物品的内容特征（类别、标签、关键词等）
	GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error)

	// GetUserPreferences 获取用户的偏好特征（喜欢的类别、标签等）
	GetUserPreferences(ctx context.Context, userID string) (map[string]float64, error)

	// GetSimilarItems 根据物品特征获取相似物品（可选，用于优化）
	GetSimilarItems(ctx context.Context, itemFeatures map[string]float64, topK int) ([]string, error)

	// ========== 矩阵分解数据 ==========

	// GetUserVector 获取用户的隐向量
	GetUserVector(ctx context.Context, userID string) ([]float64, error)

	// GetItemVector 获取物品的隐向量
	GetItemVector(ctx context.Context, itemID string) ([]float64, error)

	// GetAllItemVectors 获取所有物品的隐向量（用于在线召回）
	GetAllItemVectors(ctx context.Context) (map[string][]float64, error)

	// ========== 通用方法 ==========

	// GetAllUsers 获取所有用户 ID 列表
	GetAllUsers(ctx context.Context) ([]string, error)

	// GetAllItems 获取所有物品 ID 列表
	GetAllItems(ctx context.Context) ([]string, error)
}
