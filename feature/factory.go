package feature

import (
	"time"

	"github.com/rushteam/reckit/store"
)

// FeatureServiceFactory 是特征服务工厂，采用工厂模式创建不同类型的特征服务。
type FeatureServiceFactory struct{}

// NewFeatureServiceFactory 创建特征服务工厂
func NewFeatureServiceFactory() *FeatureServiceFactory {
	return &FeatureServiceFactory{}
}

// CreateFromStore 从 Store 创建特征服务（最常用）
func (f *FeatureServiceFactory) CreateFromStore(
	store store.Store,
	opts ...ServiceOption,
) FeatureService {
	keyPrefix := KeyPrefix{
		User:     "user:features:",
		Item:     "item:features:",
		Realtime: "realtime:features:",
	}

	provider := NewStoreFeatureProvider(store, keyPrefix)
	return NewBaseFeatureService(provider, opts...)
}

// CreateWithCache 创建带缓存的特征服务
func (f *FeatureServiceFactory) CreateWithCache(
	store store.Store,
	cacheSize int,
	cacheTTL time.Duration,
	opts ...ServiceOption,
) FeatureService {
	cache := NewMemoryFeatureCache(cacheSize, cacheTTL)
	opts = append(opts, WithCache(cache, cacheTTL))
	return f.CreateFromStore(store, opts...)
}

// CreateWithMonitor 创建带监控的特征服务
func (f *FeatureServiceFactory) CreateWithMonitor(
	store store.Store,
	maxSamples int,
	opts ...ServiceOption,
) FeatureService {
	monitor := NewMemoryFeatureMonitor(maxSamples)
	opts = append(opts, WithMonitor(monitor))
	return f.CreateFromStore(store, opts...)
}

// CreateWithFallback 创建带降级策略的特征服务
func (f *FeatureServiceFactory) CreateWithFallback(
	store store.Store,
	opts ...ServiceOption,
) FeatureService {
	fallback := NewDefaultFallbackStrategy()
	opts = append(opts, WithFallback(fallback))
	return f.CreateFromStore(store, opts...)
}

// CreateFull 创建完整的特征服务（缓存 + 监控 + 降级）
func (f *FeatureServiceFactory) CreateFull(
	store store.Store,
	cacheSize int,
	cacheTTL time.Duration,
	monitorMaxSamples int,
) FeatureService {
	cache := NewMemoryFeatureCache(cacheSize, cacheTTL)
	monitor := NewMemoryFeatureMonitor(monitorMaxSamples)
	fallback := NewDefaultFallbackStrategy()

	return NewBaseFeatureService(
		NewStoreFeatureProvider(store, KeyPrefix{}),
		WithCache(cache, cacheTTL),
		WithMonitor(monitor),
		WithFallback(fallback),
	)
}
