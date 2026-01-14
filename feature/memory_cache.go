package feature

import (
	"context"
	"sync"
	"time"
)

// MemoryFeatureCache 是内存特征缓存实现，采用 LRU 策略。
// 用于本地缓存，减少对远程特征服务的访问。
type MemoryFeatureCache struct {
	mu                sync.RWMutex
	userFeatures      map[int64]*cacheEntry
	itemFeatures      map[int64]*cacheEntry
	maxSize           int
	defaultTTL        time.Duration
	cleanupInterval   time.Duration
	cleanupTicker     *time.Ticker
	stopCleanup       chan struct{}
}

type cacheEntry struct {
	features   map[string]float64
	expireTime time.Time
	accessTime time.Time
}

// NewMemoryFeatureCache 创建内存特征缓存
func NewMemoryFeatureCache(maxSize int, defaultTTL time.Duration) *MemoryFeatureCache {
	cache := &MemoryFeatureCache{
		userFeatures:    make(map[int64]*cacheEntry),
		itemFeatures:    make(map[int64]*cacheEntry),
		maxSize:         maxSize,
		defaultTTL:      defaultTTL,
		cleanupInterval: 1 * time.Minute,
		stopCleanup:     make(chan struct{}),
	}

	// 启动清理协程
	cache.cleanupTicker = time.NewTicker(cache.cleanupInterval)
	go cache.cleanup()

	return cache
}

func (c *MemoryFeatureCache) cleanup() {
	for {
		select {
		case <-c.cleanupTicker.C:
			c.cleanExpired()
		case <-c.stopCleanup:
			c.cleanupTicker.Stop()
			return
		}
	}
}

func (c *MemoryFeatureCache) cleanExpired() {
	now := time.Now()
	c.mu.Lock()
	defer c.mu.Unlock()

	// 清理过期的用户特征
	for userID, entry := range c.userFeatures {
		if now.After(entry.expireTime) {
			delete(c.userFeatures, userID)
		}
	}

	// 清理过期的物品特征
	for itemID, entry := range c.itemFeatures {
		if now.After(entry.expireTime) {
			delete(c.itemFeatures, itemID)
		}
	}

	// 如果超过最大大小，删除最久未访问的条目
	c.evictLRU()
}

func (c *MemoryFeatureCache) evictLRU() {
	// 用户特征 LRU 清理
	if len(c.userFeatures) > c.maxSize {
		c.evictLRUFromMap(c.userFeatures)
	}

	// 物品特征 LRU 清理
	if len(c.itemFeatures) > c.maxSize {
		c.evictLRUFromMap(c.itemFeatures)
	}
}

func (c *MemoryFeatureCache) evictLRUFromMap(m map[int64]*cacheEntry) {
	if len(m) <= c.maxSize {
		return
	}

	// 找到最久未访问的条目
	var oldestKey int64
	var oldestTime time.Time
	first := true

	for key, entry := range m {
		if first || entry.accessTime.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.accessTime
			first = false
		}
	}

	if !first {
		delete(m, oldestKey)
	}
}

func (c *MemoryFeatureCache) GetUserFeatures(ctx context.Context, userID int64) (map[string]float64, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, ok := c.userFeatures[userID]
	if !ok {
		return nil, false
	}

	// 检查是否过期
	if time.Now().After(entry.expireTime) {
		return nil, false
	}

	// 更新访问时间
	entry.accessTime = time.Now()

	return entry.features, true
}

func (c *MemoryFeatureCache) SetUserFeatures(ctx context.Context, userID int64, features map[string]float64, ttl time.Duration) {
	if ttl <= 0 {
		ttl = c.defaultTTL
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// 如果超过最大大小，先清理
	if len(c.userFeatures) >= c.maxSize {
		c.evictLRUFromMap(c.userFeatures)
	}

	c.userFeatures[userID] = &cacheEntry{
		features:   features,
		expireTime: time.Now().Add(ttl),
		accessTime: time.Now(),
	}
}

func (c *MemoryFeatureCache) GetItemFeatures(ctx context.Context, itemID int64) (map[string]float64, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, ok := c.itemFeatures[itemID]
	if !ok {
		return nil, false
	}

	// 检查是否过期
	if time.Now().After(entry.expireTime) {
		return nil, false
	}

	// 更新访问时间
	entry.accessTime = time.Now()

	return entry.features, true
}

func (c *MemoryFeatureCache) SetItemFeatures(ctx context.Context, itemID int64, features map[string]float64, ttl time.Duration) {
	if ttl <= 0 {
		ttl = c.defaultTTL
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// 如果超过最大大小，先清理
	if len(c.itemFeatures) >= c.maxSize {
		c.evictLRUFromMap(c.itemFeatures)
	}

	c.itemFeatures[itemID] = &cacheEntry{
		features:   features,
		expireTime: time.Now().Add(ttl),
		accessTime: time.Now(),
	}
}

func (c *MemoryFeatureCache) InvalidateUserFeatures(ctx context.Context, userID int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.userFeatures, userID)
}

func (c *MemoryFeatureCache) InvalidateItemFeatures(ctx context.Context, itemID int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.itemFeatures, itemID)
}

func (c *MemoryFeatureCache) Clear(ctx context.Context) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.userFeatures = make(map[int64]*cacheEntry)
	c.itemFeatures = make(map[int64]*cacheEntry)
}

// Close 关闭缓存，停止清理协程
func (c *MemoryFeatureCache) Close() {
	close(c.stopCleanup)
}
