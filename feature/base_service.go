package feature

import (
	"context"
	"time"

	"github.com/rushteam/reckit/core"
)

// BaseFeatureService 是特征服务的基础实现，采用组合模式，将不同的 FeatureProvider 组合。
// 支持缓存、监控、降级等装饰器。
type BaseFeatureService struct {
	provider        FeatureProvider
	cache           FeatureCache
	monitor         FeatureMonitor
	fallback        FallbackStrategy
	enableCache     bool
	enableMonitor   bool
	enableFallback  bool
	cacheTTL        time.Duration
}

// NewBaseFeatureService 创建基础特征服务
func NewBaseFeatureService(provider FeatureProvider, opts ...ServiceOption) *BaseFeatureService {
	service := &BaseFeatureService{
		provider:       provider,
		enableCache:    false,
		enableMonitor:  false,
		enableFallback: false,
		cacheTTL:       5 * time.Minute, // 默认缓存 5 分钟
	}

	// 应用选项
	for _, opt := range opts {
		opt(service)
	}

	return service
}

// ServiceOption 是特征服务的配置选项，采用函数式选项模式。
type ServiceOption func(*BaseFeatureService)

// WithCache 启用缓存
func WithCache(cache FeatureCache, ttl time.Duration) ServiceOption {
	return func(s *BaseFeatureService) {
		s.cache = cache
		s.enableCache = true
		s.cacheTTL = ttl
	}
}

// WithMonitor 启用监控
func WithMonitor(monitor FeatureMonitor) ServiceOption {
	return func(s *BaseFeatureService) {
		s.monitor = monitor
		s.enableMonitor = true
	}
}

// WithFallback 启用降级策略
func WithFallback(fallback FallbackStrategy) ServiceOption {
	return func(s *BaseFeatureService) {
		s.fallback = fallback
		s.enableFallback = true
	}
}

func (s *BaseFeatureService) Name() string {
	return s.provider.Name()
}

func (s *BaseFeatureService) GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error) {
	// 1. 尝试从缓存获取
	if s.enableCache && s.cache != nil {
		if features, ok := s.cache.GetUserFeatures(ctx, userID); ok {
			return features, nil
		}
	}

	// 2. 从提供者获取
	features, err := s.provider.GetUserFeatures(ctx, userID)
	if err != nil {
		// 记录错误
		if s.enableMonitor && s.monitor != nil {
			s.monitor.RecordFeatureError(ctx, "user_features", err)
		}

		// 尝试降级
		if s.enableFallback && s.fallback != nil {
			// 注意：降级需要 RecommendContext，这里简化处理
			return s.fallback.GetUserFeatures(ctx, userID, nil)
		}

		return nil, err
	}

	// 3. 写入缓存
	if s.enableCache && s.cache != nil {
		s.cache.SetUserFeatures(ctx, userID, features, s.cacheTTL)
	}

	// 4. 记录监控
	if s.enableMonitor && s.monitor != nil {
		for name, value := range features {
			s.monitor.RecordFeatureUsage(ctx, name, value)
		}
	}

	return features, nil
}

func (s *BaseFeatureService) BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error) {
	if len(userIDs) == 0 {
		return make(map[string]map[string]float64), nil
	}

	result := make(map[string]map[string]float64)
	missedIDs := make([]string, 0)

	// 1. 尝试从缓存获取
	if s.enableCache && s.cache != nil {
		for _, userID := range userIDs {
			if features, ok := s.cache.GetUserFeatures(ctx, userID); ok {
				result[userID] = features
			} else {
				missedIDs = append(missedIDs, userID)
			}
		}
	} else {
		missedIDs = userIDs
	}

	// 2. 批量获取缺失的特征
	if len(missedIDs) > 0 {
		features, err := s.provider.BatchGetUserFeatures(ctx, missedIDs)
		if err != nil {
			// 记录错误
			if s.enableMonitor && s.monitor != nil {
				s.monitor.RecordFeatureError(ctx, "user_features", err)
			}

			// 尝试降级（简化处理）
			if s.enableFallback && s.fallback != nil {
				for _, userID := range missedIDs {
					if fallbackFeatures, err := s.fallback.GetUserFeatures(ctx, userID, nil); err == nil {
						result[userID] = fallbackFeatures
					}
				}
				return result, nil
			}

			return result, err
		}

		// 3. 合并结果并写入缓存
		for userID, features := range features {
			result[userID] = features
			if s.enableCache && s.cache != nil {
				s.cache.SetUserFeatures(ctx, userID, features, s.cacheTTL)
			}
			// 记录监控
			if s.enableMonitor && s.monitor != nil {
				for name, value := range features {
					s.monitor.RecordFeatureUsage(ctx, name, value)
				}
			}
		}
	}

	return result, nil
}

func (s *BaseFeatureService) GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error) {
	// 1. 尝试从缓存获取
	if s.enableCache && s.cache != nil {
		if features, ok := s.cache.GetItemFeatures(ctx, itemID); ok {
			return features, nil
		}
	}

	// 2. 从提供者获取
	features, err := s.provider.GetItemFeatures(ctx, itemID)
	if err != nil {
		// 记录错误
		if s.enableMonitor && s.monitor != nil {
			s.monitor.RecordFeatureError(ctx, "item_features", err)
		}

		// 尝试降级
		if s.enableFallback && s.fallback != nil {
			return s.fallback.GetItemFeatures(ctx, itemID, nil)
		}

		return nil, err
	}

	// 3. 写入缓存
	if s.enableCache && s.cache != nil {
		s.cache.SetItemFeatures(ctx, itemID, features, s.cacheTTL)
	}

	// 4. 记录监控
	if s.enableMonitor && s.monitor != nil {
		for name, value := range features {
			s.monitor.RecordFeatureUsage(ctx, name, value)
		}
	}

	return features, nil
}

func (s *BaseFeatureService) BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error) {
	if len(itemIDs) == 0 {
		return make(map[string]map[string]float64), nil
	}

	result := make(map[string]map[string]float64)
	missedIDs := make([]string, 0)

	// 1. 尝试从缓存获取
	if s.enableCache && s.cache != nil {
		for _, itemID := range itemIDs {
			if features, ok := s.cache.GetItemFeatures(ctx, itemID); ok {
				result[itemID] = features
			} else {
				missedIDs = append(missedIDs, itemID)
			}
		}
	} else {
		missedIDs = itemIDs
	}

	// 2. 批量获取缺失的特征
	if len(missedIDs) > 0 {
		features, err := s.provider.BatchGetItemFeatures(ctx, missedIDs)
		if err != nil {
			// 记录错误
			if s.enableMonitor && s.monitor != nil {
				s.monitor.RecordFeatureError(ctx, "item_features", err)
			}

			// 尝试降级
			if s.enableFallback && s.fallback != nil {
				for _, itemID := range missedIDs {
					if fallbackFeatures, err := s.fallback.GetItemFeatures(ctx, itemID, nil); err == nil {
						result[itemID] = fallbackFeatures
					}
				}
				return result, nil
			}

			return result, err
		}

		// 3. 合并结果并写入缓存
		for itemID, features := range features {
			result[itemID] = features
			if s.enableCache && s.cache != nil {
				s.cache.SetItemFeatures(ctx, itemID, features, s.cacheTTL)
			}
			// 记录监控
			if s.enableMonitor && s.monitor != nil {
				for name, value := range features {
					s.monitor.RecordFeatureUsage(ctx, name, value)
				}
			}
		}
	}

	return result, nil
}

// 确保 BaseFeatureService 实现了 core.FeatureService 接口
var _ core.FeatureService = (*BaseFeatureService)(nil)

func (s *BaseFeatureService) Close(ctx context.Context) error {
	// 清理资源
	if s.cache != nil {
		s.cache.Clear(context.Background())
	}
	return nil
}
