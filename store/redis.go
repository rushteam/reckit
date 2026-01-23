package store

import (
	"context"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/rushteam/reckit/core"
)

// RedisStore 是 Redis 实现的 KeyValueStore，支持所有 Redis 数据结构操作。
// 生产环境常用，支持持久化、集群、哨兵等。
type RedisStore struct {
	client *redis.Client
}

func NewRedisStore(addr string, db int) (*RedisStore, error) {
	client := redis.NewClient(&redis.Options{
		Addr: addr,
		DB:   db,
	})
	if err := client.Ping(context.Background()).Err(); err != nil {
		return nil, err
	}
	return &RedisStore{client: client}, nil
}

func (r *RedisStore) Name() string { return "redis" }

func (r *RedisStore) Get(ctx context.Context, key string) ([]byte, error) {
	val, err := r.client.Get(ctx, key).Bytes()
	if err == redis.Nil {
		return nil, core.ErrStoreNotFound
	}
	return val, err
}

func (r *RedisStore) Set(ctx context.Context, key string, value []byte, ttl ...int) error {
	var expiration time.Duration
	if len(ttl) > 0 && ttl[0] > 0 {
		expiration = time.Duration(ttl[0]) * time.Second
	}
	return r.client.Set(ctx, key, value, expiration).Err()
}

func (r *RedisStore) Delete(ctx context.Context, key string) error {
	return r.client.Del(ctx, key).Err()
}

func (r *RedisStore) BatchGet(ctx context.Context, keys []string) (map[string][]byte, error) {
	if len(keys) == 0 {
		return make(map[string][]byte), nil
	}

	vals, err := r.client.MGet(ctx, keys...).Result()
	if err != nil {
		return nil, err
	}

	result := make(map[string][]byte, len(keys))
	for i, k := range keys {
		if vals[i] != nil {
			if s, ok := vals[i].(string); ok {
				result[k] = []byte(s)
			}
		}
	}
	return result, nil
}

func (r *RedisStore) BatchSet(ctx context.Context, kvs map[string][]byte, ttl ...int) error {
	pipe := r.client.Pipeline()
	var expiration time.Duration
	if len(ttl) > 0 && ttl[0] > 0 {
		expiration = time.Duration(ttl[0]) * time.Second
	}

	for k, v := range kvs {
		pipe.Set(ctx, k, v, expiration)
	}
	_, err := pipe.Exec(ctx)
	return err
}

func (r *RedisStore) ZAdd(ctx context.Context, key string, score float64, member string) error {
	return r.client.ZAdd(ctx, key, redis.Z{Score: score, Member: member}).Err()
}

func (r *RedisStore) ZRange(ctx context.Context, key string, start, stop int64) ([]string, error) {
	return r.client.ZRevRange(ctx, key, start, stop).Result()
}

func (r *RedisStore) ZScore(ctx context.Context, key string, member string) (float64, error) {
	score, err := r.client.ZScore(ctx, key, member).Result()
	if err == redis.Nil {
		return 0, core.ErrStoreNotFound
	}
	return score, err
}

func (r *RedisStore) HGet(ctx context.Context, key, field string) ([]byte, error) {
	val, err := r.client.HGet(ctx, key, field).Bytes()
	if err == redis.Nil {
		return nil, core.ErrStoreNotFound
	}
	return val, err
}

func (r *RedisStore) HSet(ctx context.Context, key, field string, value []byte) error {
	return r.client.HSet(ctx, key, field, value).Err()
}

func (r *RedisStore) HGetAll(ctx context.Context, key string) (map[string][]byte, error) {
	vals, err := r.client.HGetAll(ctx, key).Result()
	if err != nil {
		return nil, err
	}
	result := make(map[string][]byte, len(vals))
	for k, v := range vals {
		result[k] = []byte(v)
	}
	return result, nil
}

func (r *RedisStore) Close() error {
	return r.client.Close()
}

// 确保 RedisStore 实现了 core.Store 和 core.KeyValueStore 接口
var _ core.Store = (*RedisStore)(nil)
var _ core.KeyValueStore = (*RedisStore)(nil)
