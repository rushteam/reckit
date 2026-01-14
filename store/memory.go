package store

import (
	"context"
	"sort"
	"sync"
	"time"
)

// MemoryStore 是内存实现的 Store，用于测试/开发/原型。
// 支持 TTL（过期时间），但进程重启后数据丢失。
type MemoryStore struct {
	mu    sync.RWMutex
	data  map[string]*entry
	ttl   map[string]time.Time
	zsets map[string]map[string]float64 // zset key -> member -> score
	clean *time.Ticker
}

type entry struct {
	value []byte
	ttl   *time.Time
}

func NewMemoryStore() *MemoryStore {
	ms := &MemoryStore{
		data:  make(map[string]*entry),
		ttl:   make(map[string]time.Time),
		zsets: make(map[string]map[string]float64),
		clean: time.NewTicker(10 * time.Second),
	}
	go ms.cleanup()
	return ms
}

func (m *MemoryStore) Name() string { return "memory" }

func (m *MemoryStore) Get(ctx context.Context, key string) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	e, ok := m.data[key]
	if !ok {
		return nil, ErrNotFound
	}
	if e.ttl != nil && time.Now().After(*e.ttl) {
		return nil, ErrNotFound
	}
	return e.value, nil
}

func (m *MemoryStore) Set(ctx context.Context, key string, value []byte, ttl ...int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	e := &entry{value: value}
	if len(ttl) > 0 && ttl[0] > 0 {
		expire := time.Now().Add(time.Duration(ttl[0]) * time.Second)
		e.ttl = &expire
		m.ttl[key] = expire
	}
	m.data[key] = e
	return nil
}

func (m *MemoryStore) Delete(ctx context.Context, key string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.data, key)
	delete(m.ttl, key)
	return nil
}

func (m *MemoryStore) BatchGet(ctx context.Context, keys []string) (map[string][]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	result := make(map[string][]byte, len(keys))
	now := time.Now()
	for _, k := range keys {
		e, ok := m.data[k]
		if !ok {
			continue
		}
		if e.ttl != nil && now.After(*e.ttl) {
			continue
		}
		result[k] = e.value
	}
	return result, nil
}

func (m *MemoryStore) BatchSet(ctx context.Context, kvs map[string][]byte, ttl ...int) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	var expire *time.Time
	if len(ttl) > 0 && ttl[0] > 0 {
		t := time.Now().Add(time.Duration(ttl[0]) * time.Second)
		expire = &t
	}

	for k, v := range kvs {
		e := &entry{value: v, ttl: expire}
		m.data[k] = e
		if expire != nil {
			m.ttl[k] = *expire
		}
	}
	return nil
}

func (m *MemoryStore) Close() error {
	if m.clean != nil {
		m.clean.Stop()
	}
	return nil
}

func (m *MemoryStore) cleanup() {
	for range m.clean.C {
		m.mu.Lock()
		now := time.Now()
		for k, expire := range m.ttl {
			if now.After(expire) {
				delete(m.data, k)
				delete(m.ttl, k)
			}
		}
		m.mu.Unlock()
	}
}

// KeyValueStore 扩展方法（MemoryStore 也实现 KeyValueStore 接口）

var _ KeyValueStore = (*MemoryStore)(nil)

func (m *MemoryStore) ZAdd(ctx context.Context, key string, score float64, member string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.zsets[key] == nil {
		m.zsets[key] = make(map[string]float64)
	}
	m.zsets[key][member] = score
	return nil
}

func (m *MemoryStore) ZRange(ctx context.Context, key string, start, stop int64) ([]string, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	zset, ok := m.zsets[key]
	if !ok || len(zset) == 0 {
		return nil, nil
	}

	// 转换为 slice 并按 score 降序排序
	type pair struct {
		member string
		score  float64
	}
	pairs := make([]pair, 0, len(zset))
	for m, s := range zset {
		pairs = append(pairs, pair{member: m, score: s})
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score > pairs[j].score
	})

	// 处理范围
	if start < 0 {
		start = 0
	}
	if stop < 0 || stop >= int64(len(pairs)) {
		stop = int64(len(pairs)) - 1
	}
	if start > stop {
		return nil, nil
	}

	result := make([]string, 0, stop-start+1)
	for i := start; i <= stop && i < int64(len(pairs)); i++ {
		result = append(result, pairs[i].member)
	}
	return result, nil
}

func (m *MemoryStore) ZScore(ctx context.Context, key string, member string) (float64, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	zset, ok := m.zsets[key]
	if !ok {
		return 0, ErrNotFound
	}
	score, ok := zset[member]
	if !ok {
		return 0, ErrNotFound
	}
	return score, nil
}

func (m *MemoryStore) HGet(ctx context.Context, key, field string) ([]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	hkey := "hash:" + key + ":" + field
	e, ok := m.data[hkey]
	if !ok {
		return nil, ErrNotFound
	}
	if e.ttl != nil && time.Now().After(*e.ttl) {
		return nil, ErrNotFound
	}
	return e.value, nil
}

func (m *MemoryStore) HSet(ctx context.Context, key, field string, value []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	hkey := "hash:" + key + ":" + field
	m.data[hkey] = &entry{value: value}
	return nil
}

func (m *MemoryStore) HGetAll(ctx context.Context, key string) (map[string][]byte, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	prefix := "hash:" + key + ":"
	result := make(map[string][]byte)
	now := time.Now()
	for k, e := range m.data {
		if len(k) > len(prefix) && k[:len(prefix)] == prefix {
			if e.ttl != nil && now.After(*e.ttl) {
				continue
			}
			field := k[len(prefix):]
			result[field] = e.value
		}
	}
	return result, nil
}
