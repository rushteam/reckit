package store

import (
	"context"
	"errors"
)

var (
	ErrNotFound = errors.New("store: key not found")
)

// Store 是 Reckit 的存储抽象接口，支持多种后端（Redis、MySQL、ES、内存等）。
// 统一用于召回、特征读取、缓存等场景。
type Store interface {
	// Name 返回存储后端名称（用于日志/监控）。
	Name() string

	// Get 读取单个 key 的值。
	Get(ctx context.Context, key string) ([]byte, error)

	// Set 写入单个 key-value。
	Set(ctx context.Context, key string, value []byte, ttl ...int) error

	// Delete 删除单个 key。
	Delete(ctx context.Context, key string) error

	// BatchGet 批量读取（推荐系统常用，减少网络往返）。
	BatchGet(ctx context.Context, keys []string) (map[string][]byte, error)

	// BatchSet 批量写入。
	BatchSet(ctx context.Context, kvs map[string][]byte, ttl ...int) error

	// Close 关闭连接/释放资源。
	Close() error
}

// KeyValueStore 是 Store 的扩展接口，支持更丰富的 KV 操作（例如 Redis 的 SortedSet、List）。
// 如果后端不支持某些操作，可返回 ErrNotSupported。
type KeyValueStore interface {
	Store

	// ZAdd 向有序集合添加成员（用于热门排序、时间线等）。
	ZAdd(ctx context.Context, key string, score float64, member string) error

	// ZRange 按分数范围获取有序集合成员（降序，用于 TopN 召回）。
	ZRange(ctx context.Context, key string, start, stop int64) ([]string, error)

	// ZScore 获取成员的分数。
	ZScore(ctx context.Context, key string, member string) (float64, error)

	// HGet 读取 Hash 字段（用于物品元数据、特征）。
	HGet(ctx context.Context, key, field string) ([]byte, error)

	// HSet 写入 Hash 字段。
	HSet(ctx context.Context, key, field string, value []byte) error

	// HGetAll 读取整个 Hash（用于批量特征）。
	HGetAll(ctx context.Context, key string) (map[string][]byte, error)
}

var ErrNotSupported = errors.New("store: operation not supported")
