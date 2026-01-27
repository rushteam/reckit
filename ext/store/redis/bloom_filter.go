package redis

import (
	"bytes"
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/bits-and-blooms/bloom/v3"
	"github.com/redis/go-redis/v9"
	"github.com/rushteam/reckit/filter"
)

// RedisBloomFilterChecker 是基于 Redis 和 bits-and-blooms/bloom 的布隆过滤器检查器。
// 实现了 filter.BloomFilterChecker 接口，用于曝光过滤器的布隆过滤器检查。
//
// 使用方式：
//   checker := redis.NewRedisBloomFilterChecker(redisClient, 1000000, 0.01)
//   storeAdapter := filter.NewStoreAdapterWithBloomFilter(store, checker)
//   exposedFilter := filter.NewExposedFilter(storeAdapter, "user:exposed", 7*24*3600, 30)
//
// 确保实现了 filter.BloomFilterChecker 接口
var _ filter.BloomFilterChecker = (*RedisBloomFilterChecker)(nil)

type RedisBloomFilterChecker struct {
	client *redis.Client

	// 布隆过滤器参数
	// capacity 是预期容量（元素数量）
	capacity uint
	// falsePositiveRate 是期望的误判率（例如 0.01 表示 1%）
	falsePositiveRate float64

	// 本地缓存，避免频繁从 Redis 读取和反序列化
	cache map[string]*bloom.BloomFilter
	mu    sync.RWMutex
}

// NewRedisBloomFilterChecker 创建一个新的 Redis 布隆过滤器检查器。
//
// 参数：
//   - store: RedisStore 实例
//   - capacity: 预期容量（元素数量），例如 1000000 表示预期存储 100 万个元素
//   - falsePositiveRate: 期望的误判率，例如 0.01 表示 1% 的误判率
//
// 示例：
//   store, _ := NewRedisStore("localhost:6379", 0)
//   checker := NewRedisBloomFilterChecker(store, 1000000, 0.01)
func NewRedisBloomFilterChecker(store *RedisStore, capacity uint, falsePositiveRate float64) *RedisBloomFilterChecker {
	return &RedisBloomFilterChecker{
		client:            store.GetClient(),
		capacity:          capacity,
		falsePositiveRate: falsePositiveRate,
		cache:             make(map[string]*bloom.BloomFilter),
	}
}

// NewRedisBloomFilterCheckerWithClient 使用 *redis.Client 创建布隆过滤器检查器（高级用法）。
// 如果已有 *redis.Client 实例，可以使用此方法。
func NewRedisBloomFilterCheckerWithClient(client *redis.Client, capacity uint, falsePositiveRate float64) *RedisBloomFilterChecker {
	return &RedisBloomFilterChecker{
		client:            client,
		capacity:          capacity,
		falsePositiveRate: falsePositiveRate,
		cache:             make(map[string]*bloom.BloomFilter),
	}
}

// CheckInBloomFilter 检查 itemID 是否在指定 key 的布隆过滤器中。
// 实现了 filter.BloomFilterChecker 接口。
//
// 参数：
//   - ctx: 上下文
//   - key: 布隆过滤器的 Redis key，格式为 {keyPrefix}:bloom:{userID}:{date}
//   - itemID: 要检查的物品 ID
//
// 返回：
//   - bool: true 表示可能在布隆过滤器中（存在误判可能），false 表示一定不在
//   - error: 错误信息
func (r *RedisBloomFilterChecker) CheckInBloomFilter(ctx context.Context, key string, itemID string) (bool, error) {
	// 1. 尝试从本地缓存获取
	r.mu.RLock()
	cached, exists := r.cache[key]
	r.mu.RUnlock()

	if exists && cached != nil {
		// 使用缓存的布隆过滤器检查
		return cached.Test([]byte(itemID)), nil
	}

	// 2. 从 Redis 读取布隆过滤器数据
	data, err := r.client.Get(ctx, key).Bytes()
	if err == redis.Nil {
		// 布隆过滤器不存在，表示一定不在
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("failed to get bloom filter from redis: %w", err)
	}

	// 3. 反序列化布隆过滤器
	bf := bloom.NewWithEstimates(r.capacity, r.falsePositiveRate)
	_, err = bf.ReadFrom(bytes.NewReader(data))
	if err != nil {
		return false, fmt.Errorf("failed to deserialize bloom filter: %w", err)
	}

	// 4. 缓存布隆过滤器（可选，避免频繁反序列化）
	r.mu.Lock()
	r.cache[key] = bf
	r.mu.Unlock()

	// 5. 检查 itemID 是否在布隆过滤器中
	return bf.Test([]byte(itemID)), nil
}

// AddToBloomFilter 将 itemID 添加到指定 key 的布隆过滤器中。
// 这是一个辅助方法，用于数据写入场景（例如曝光数据收集）。
//
// 参数：
//   - ctx: 上下文
//   - key: 布隆过滤器的 Redis key
//   - itemID: 要添加的物品 ID
//   - ttl: 过期时间（秒），0 表示不过期
//
// 返回：
//   - error: 错误信息
func (r *RedisBloomFilterChecker) AddToBloomFilter(ctx context.Context, key string, itemID string, ttl int) error {
	// 1. 尝试从本地缓存获取
	r.mu.RLock()
	cached, exists := r.cache[key]
	r.mu.RUnlock()

	var bf *bloom.BloomFilter
	if exists && cached != nil {
		bf = cached
	} else {
		// 2. 从 Redis 读取或创建新的布隆过滤器
		data, err := r.client.Get(ctx, key).Bytes()
		if err == redis.Nil {
			// 不存在，创建新的布隆过滤器
			bf = bloom.NewWithEstimates(r.capacity, r.falsePositiveRate)
		} else if err != nil {
			return fmt.Errorf("failed to get bloom filter from redis: %w", err)
		} else {
			// 反序列化现有的布隆过滤器
			bf = bloom.NewWithEstimates(r.capacity, r.falsePositiveRate)
			_, err = bf.ReadFrom(bytes.NewReader(data))
			if err != nil {
				return fmt.Errorf("failed to deserialize bloom filter: %w", err)
			}
		}
	}

	// 3. 添加 itemID 到布隆过滤器
	bf.Add([]byte(itemID))

	// 4. 序列化并写回 Redis
	var buf bytes.Buffer
	_, err := bf.WriteTo(&buf)
	if err != nil {
		return fmt.Errorf("failed to serialize bloom filter: %w", err)
	}

	var expiration time.Duration
	if ttl > 0 {
		expiration = time.Duration(ttl) * time.Second
	}

	err = r.client.Set(ctx, key, buf.Bytes(), expiration).Err()
	if err != nil {
		return fmt.Errorf("failed to save bloom filter to redis: %w", err)
	}

	// 5. 更新本地缓存
	r.mu.Lock()
	r.cache[key] = bf
	r.mu.Unlock()

	return nil
}

// BatchAddToBloomFilter 批量将 itemIDs 添加到指定 key 的布隆过滤器中。
// 这是一个辅助方法，用于批量数据写入场景。
//
// 参数：
//   - ctx: 上下文
//   - key: 布隆过滤器的 Redis key
//   - itemIDs: 要添加的物品 ID 列表
//   - ttl: 过期时间（秒），0 表示不过期
//
// 返回：
//   - error: 错误信息
func (r *RedisBloomFilterChecker) BatchAddToBloomFilter(ctx context.Context, key string, itemIDs []string, ttl int) error {
	// 1. 尝试从本地缓存获取
	r.mu.RLock()
	cached, exists := r.cache[key]
	r.mu.RUnlock()

	var bf *bloom.BloomFilter
	if exists && cached != nil {
		bf = cached
	} else {
		// 2. 从 Redis 读取或创建新的布隆过滤器
		data, err := r.client.Get(ctx, key).Bytes()
		if err == redis.Nil {
			// 不存在，创建新的布隆过滤器
			bf = bloom.NewWithEstimates(r.capacity, r.falsePositiveRate)
		} else if err != nil {
			return fmt.Errorf("failed to get bloom filter from redis: %w", err)
		} else {
			// 反序列化现有的布隆过滤器
			bf = bloom.NewWithEstimates(r.capacity, r.falsePositiveRate)
			_, err = bf.ReadFrom(bytes.NewReader(data))
			if err != nil {
				return fmt.Errorf("failed to deserialize bloom filter: %w", err)
			}
		}
	}

	// 3. 批量添加 itemIDs 到布隆过滤器
	for _, itemID := range itemIDs {
		bf.Add([]byte(itemID))
	}

	// 4. 序列化并写回 Redis
	var buf bytes.Buffer
	_, err := bf.WriteTo(&buf)
	if err != nil {
		return fmt.Errorf("failed to serialize bloom filter: %w", err)
	}

	var expiration time.Duration
	if ttl > 0 {
		expiration = time.Duration(ttl) * time.Second
	}

	err = r.client.Set(ctx, key, buf.Bytes(), expiration).Err()
	if err != nil {
		return fmt.Errorf("failed to save bloom filter to redis: %w", err)
	}

	// 5. 更新本地缓存
	r.mu.Lock()
	r.cache[key] = bf
	r.mu.Unlock()

	return nil
}

// ClearCache 清除本地缓存。
// 当需要强制从 Redis 重新加载布隆过滤器时使用。
func (r *RedisBloomFilterChecker) ClearCache() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.cache = make(map[string]*bloom.BloomFilter)
}

// ClearCacheKey 清除指定 key 的本地缓存。
func (r *RedisBloomFilterChecker) ClearCacheKey(key string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.cache, key)
}
