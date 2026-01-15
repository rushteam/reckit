package feature

import (
	"context"
	"errors"
	"time"

	"reckit/core"
)

var (
	// ErrFeatureNotFound 特征未找到
	ErrFeatureNotFound = errors.New("feature: feature not found")
	// ErrFeatureServiceUnavailable 特征服务不可用
	ErrFeatureServiceUnavailable = errors.New("feature: service unavailable")
)

// FeatureService 是特征服务的统一接口，提供用户特征、物品特征、实时特征的获取能力。
// 采用策略模式，支持多种实现（Redis、HTTP、Memory等）。
//
// ID 类型：使用 string（通用，支持所有 ID 格式）
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
	BatchGetRealtimeFeatures(ctx context.Context, pairs []UserItemPair) (map[UserItemPair]map[string]float64, error)

	// Close 关闭特征服务，释放资源
	Close() error
}

// UserItemPair 用户-物品对，用于实时特征查询
type UserItemPair struct {
	UserID string
	ItemID string
}

// FeatureProvider 是特征提供者的抽象接口，采用策略模式。
// 不同的特征源（Redis、HTTP、Memory）实现此接口。
//
// ID 类型：使用 string（通用，支持所有 ID 格式）
type FeatureProvider interface {
	// Name 返回提供者名称
	Name() string

	// GetUserFeatures 获取用户特征
	GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error)

	// BatchGetUserFeatures 批量获取用户特征
	BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error)

	// GetItemFeatures 获取物品特征
	GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error)

	// BatchGetItemFeatures 批量获取物品特征
	BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error)

	// GetRealtimeFeatures 获取实时特征
	GetRealtimeFeatures(ctx context.Context, userID, itemID string) (map[string]float64, error)

	// BatchGetRealtimeFeatures 批量获取实时特征
	BatchGetRealtimeFeatures(ctx context.Context, pairs []UserItemPair) (map[UserItemPair]map[string]float64, error)
}

// FeatureStore 是特征存储的抽象接口，用于特征数据的持久化和读取。
// 与 store.Store 不同，FeatureStore 专门用于特征数据，支持特征版本管理。
//
// ID 类型：使用 string（通用，支持所有 ID 格式）
type FeatureStore interface {
	// Name 返回存储名称
	Name() string

	// GetUserFeatures 从存储读取用户特征
	GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error)

	// BatchGetUserFeatures 批量读取用户特征
	BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error)

	// GetItemFeatures 从存储读取物品特征
	GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error)

	// BatchGetItemFeatures 批量读取物品特征
	BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error)

	// GetRealtimeFeatures 从存储读取实时特征
	GetRealtimeFeatures(ctx context.Context, userID, itemID string) (map[string]float64, error)

	// BatchGetRealtimeFeatures 批量读取实时特征
	BatchGetRealtimeFeatures(ctx context.Context, pairs []UserItemPair) (map[UserItemPair]map[string]float64, error)

	// SetUserFeatures 写入用户特征（可选，用于特征更新）
	SetUserFeatures(ctx context.Context, userID string, features map[string]float64, ttl time.Duration) error

	// SetItemFeatures 写入物品特征（可选，用于特征更新）
	SetItemFeatures(ctx context.Context, itemID string, features map[string]float64, ttl time.Duration) error
}

// FeatureMonitor 是特征监控接口，用于监控特征质量、分布、缺失率等。
type FeatureMonitor interface {
	// RecordFeatureUsage 记录特征使用情况
	RecordFeatureUsage(ctx context.Context, featureName string, value float64)

	// RecordFeatureMissing 记录特征缺失
	RecordFeatureMissing(ctx context.Context, featureName string, entityType string, entityID string)

	// RecordFeatureError 记录特征获取错误
	RecordFeatureError(ctx context.Context, featureName string, err error)

	// GetFeatureStats 获取特征统计信息
	GetFeatureStats(ctx context.Context, featureName string) (*FeatureStats, error)
}

// FeatureStats 特征统计信息
type FeatureStats struct {
	FeatureName    string
	UsageCount     int64
	MissingCount   int64
	ErrorCount     int64
	Mean           float64
	Std            float64
	Min            float64
	Max            float64
	P50            float64
	P95            float64
	P99            float64
	LastUpdateTime time.Time
}

// FeatureCache 是特征缓存接口，采用装饰器模式，为特征服务添加缓存能力。
//
// ID 类型：使用 string（通用，支持所有 ID 格式）
type FeatureCache interface {
	// GetUserFeatures 从缓存获取用户特征
	GetUserFeatures(ctx context.Context, userID string) (map[string]float64, bool)

	// SetUserFeatures 设置用户特征缓存
	SetUserFeatures(ctx context.Context, userID string, features map[string]float64, ttl time.Duration)

	// GetItemFeatures 从缓存获取物品特征
	GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, bool)

	// SetItemFeatures 设置物品特征缓存
	SetItemFeatures(ctx context.Context, itemID string, features map[string]float64, ttl time.Duration)

	// InvalidateUserFeatures 失效用户特征缓存
	InvalidateUserFeatures(ctx context.Context, userID string)

	// InvalidateItemFeatures 失效物品特征缓存
	InvalidateItemFeatures(ctx context.Context, itemID string)

	// Clear 清空所有缓存
	Clear(ctx context.Context)
}

// FallbackStrategy 是降级策略接口，当特征服务不可用时提供降级方案。
//
// ID 类型：使用 string（通用，支持所有 ID 格式）
type FallbackStrategy interface {
	// GetUserFeatures 获取用户特征（降级方案）
	GetUserFeatures(ctx context.Context, userID string, rctx *core.RecommendContext) (map[string]float64, error)

	// GetItemFeatures 获取物品特征（降级方案）
	GetItemFeatures(ctx context.Context, itemID string, item *core.Item) (map[string]float64, error)

	// GetRealtimeFeatures 获取实时特征（降级方案）
	GetRealtimeFeatures(ctx context.Context, userID, itemID string, rctx *core.RecommendContext, item *core.Item) (map[string]float64, error)
}
