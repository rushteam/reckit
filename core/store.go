package core

import "context"

// Store 是存储的领域接口。
//
// 设计原则：
//   - 定义在领域层（core），由基础设施层（store）实现
//   - 遵循依赖倒置原则：领域层定义接口，基础设施层实现接口
//   - 避免循环依赖：领域层不依赖基础设施层
//
// 使用场景：
//   - 召回数据存储：协同过滤、内容推荐、矩阵分解等
//   - 特征存储：用户特征、物品特征
//   - 缓存：特征缓存、结果缓存
//
// 实现：
//   - store.MemoryStore 实现此接口
//   - store.RedisStore 实现此接口
//   - 其他存储后端（MySQL、ES 等）也可以实现此接口
type Store interface {
	// Name 返回存储后端名称（用于日志/监控）
	Name() string

	// Get 读取单个 key 的值
	Get(ctx context.Context, key string) ([]byte, error)

	// Set 写入单个 key-value
	Set(ctx context.Context, key string, value []byte, ttl ...int) error

	// Delete 删除单个 key
	Delete(ctx context.Context, key string) error

	// BatchGet 批量读取（推荐系统常用，减少网络往返）
	BatchGet(ctx context.Context, keys []string) (map[string][]byte, error)

	// BatchSet 批量写入
	BatchSet(ctx context.Context, kvs map[string][]byte, ttl ...int) error

	// Close 关闭连接/释放资源
	Close() error
}

// KeyValueStore 是 Store 的扩展接口，支持更丰富的 KV 操作。
//
// 扩展功能：
//   - 有序集合（SortedSet）：用于热门排序、时间线等
//   - 哈希表（Hash）：用于物品元数据、特征等
//
// 如果后端不支持某些操作，可返回 ErrStoreNotSupported。
type KeyValueStore interface {
	Store

	// ZAdd 向有序集合添加成员（用于热门排序、时间线等）
	ZAdd(ctx context.Context, key string, score float64, member string) error

	// ZRange 按分数范围获取有序集合成员（降序，用于 TopN 召回）
	ZRange(ctx context.Context, key string, start, stop int64) ([]string, error)

	// ZScore 获取成员的分数
	ZScore(ctx context.Context, key string, member string) (float64, error)

	// HGet 读取 Hash 字段（用于物品元数据、特征）
	HGet(ctx context.Context, key, field string) ([]byte, error)

	// HSet 写入 Hash 字段
	HSet(ctx context.Context, key, field string, value []byte) error

	// HGetAll 读取整个 Hash（用于批量特征）
	HGetAll(ctx context.Context, key string) (map[string][]byte, error)
}

// Store 错误定义（使用统一的 DomainError）
var (
	// ErrStoreNotFound 表示 key 不存在
	ErrStoreNotFound = NewDomainError(ModuleStore, ErrorCodeNotFound, "store: key not found")

	// ErrStoreNotSupported 表示操作不支持
	ErrStoreNotSupported = NewDomainError(ModuleStore, ErrorCodeNotSupported, "store: operation not supported")
)

// IsStoreNotFound 检查错误是否为 key 不存在（使用统一的错误检查）
func IsStoreNotFound(err error) bool {
	if err == nil {
		return false
	}
	domainErr := GetDomainError(err)
	if domainErr != nil && domainErr.Module == ModuleStore {
		return domainErr.Code == ErrorCodeNotFound
	}
	return false
}

// IsStoreNotSupported 检查错误是否为操作不支持（使用统一的错误检查）
func IsStoreNotSupported(err error) bool {
	if err == nil {
		return false
	}
	domainErr := GetDomainError(err)
	if domainErr != nil && domainErr.Module == ModuleStore {
		return domainErr.Code == ErrorCodeNotSupported
	}
	return false
}
