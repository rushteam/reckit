package filter

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// BlacklistProvider 黑名单数据源接口。
// 不同的 Provider 可对接不同数据源（Redis SET、场景配置 JSON、数据库等），
// 由 CompositeBlacklistFilter 统一合并。
type BlacklistProvider interface {
	// GetBlacklistIDs 返回当前请求应过滤的 item ID 列表。
	// 实现应自行处理内部错误（如 Redis 超时），对于不可用场景返回空列表即可。
	GetBlacklistIDs(ctx context.Context, rctx *core.RecommendContext) ([]string, error)
}

// BlacklistProviderFunc 函数适配器，方便用闭包快速构造 BlacklistProvider。
type BlacklistProviderFunc func(ctx context.Context, rctx *core.RecommendContext) ([]string, error)

func (f BlacklistProviderFunc) GetBlacklistIDs(ctx context.Context, rctx *core.RecommendContext) ([]string, error) {
	return f(ctx, rctx)
}

// CompositeBlacklistFilter 组合式黑名单过滤器。
// 合并多个 BlacklistProvider 的结果，对匹配的 item 执行过滤。
//
// 示例：
//
//	f := filter.NewCompositeBlacklistFilter(
//	    filter.StoreBlacklistProvider(redisStore, "ban:items"),
//	    filter.BlacklistProviderFunc(func(ctx context.Context, rctx *core.RecommendContext) ([]string, error) {
//	        // 从场景配置读取动态黑名单
//	        cfg, _ := core.ExtensionAs[*SceneConfig](rctx, "scene_config")
//	        if cfg != nil {
//	            return cfg.Blacklist, nil
//	        }
//	        return nil, nil
//	    }),
//	)
type CompositeBlacklistFilter struct {
	Providers []BlacklistProvider
}

// NewCompositeBlacklistFilter 创建组合黑名单过滤器。
func NewCompositeBlacklistFilter(providers ...BlacklistProvider) *CompositeBlacklistFilter {
	return &CompositeBlacklistFilter{Providers: providers}
}

func (f *CompositeBlacklistFilter) Name() string {
	return "filter.composite_blacklist"
}

func (f *CompositeBlacklistFilter) ShouldFilter(
	ctx context.Context,
	rctx *core.RecommendContext,
	item *core.Item,
) (bool, error) {
	if item == nil {
		return true, nil
	}
	banSet := f.collectBanSet(ctx, rctx)
	_, banned := banSet[item.ID]
	return banned, nil
}

// FilterBatch 批量过滤，一次合并所有 Provider 数据，O(N) 遍历。
func (f *CompositeBlacklistFilter) FilterBatch(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(items) == 0 {
		return items, nil
	}

	banSet := f.collectBanSet(ctx, rctx)
	if len(banSet) == 0 {
		return items, nil
	}

	out := make([]*core.Item, 0, len(items))
	for _, item := range items {
		if item == nil {
			continue
		}
		if _, banned := banSet[item.ID]; !banned {
			out = append(out, item)
		}
	}
	return out, nil
}

func (f *CompositeBlacklistFilter) collectBanSet(ctx context.Context, rctx *core.RecommendContext) map[string]struct{} {
	var totalSize int
	results := make([][]string, 0, len(f.Providers))
	for _, p := range f.Providers {
		ids, err := p.GetBlacklistIDs(ctx, rctx)
		if err != nil || len(ids) == 0 {
			continue
		}
		results = append(results, ids)
		totalSize += len(ids)
	}

	if totalSize == 0 {
		return nil
	}

	banSet := make(map[string]struct{}, totalSize)
	for _, ids := range results {
		for _, id := range ids {
			banSet[id] = struct{}{}
		}
	}
	return banSet
}

// StoreBlacklistProvider 基于 BlacklistStore 的 Provider 适配器。
// 从指定 key 读取静态黑名单列表。
func StoreBlacklistProvider(store BlacklistStore, key string) BlacklistProvider {
	return BlacklistProviderFunc(func(ctx context.Context, _ *core.RecommendContext) ([]string, error) {
		if store == nil || key == "" {
			return nil, nil
		}
		ids, err := store.GetBlacklist(ctx, key)
		if err != nil {
			return nil, nil
		}
		return ids, nil
	})
}

// StaticBlacklistProvider 基于静态 ID 列表的 Provider。
func StaticBlacklistProvider(ids []string) BlacklistProvider {
	return BlacklistProviderFunc(func(_ context.Context, _ *core.RecommendContext) ([]string, error) {
		return ids, nil
	})
}
