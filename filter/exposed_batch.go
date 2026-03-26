package filter

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// BatchExposureChecker 提供批量曝光判定能力。
// 返回 map[itemID]bool，true 表示该 item 已曝光（应过滤）。
type BatchExposureChecker interface {
	CheckExposedBatch(
		ctx context.Context,
		userID string,
		itemIDs []string,
		keyPrefix string,
		timeWindow int64,
		dayWindow int,
	) (map[string]bool, error)
}

// BatchExposedFilter 是已曝光过滤器的批量版本，优先用于高 QPS 场景。
// 它会在一次调用中判断整批候选，避免逐条 IO。
type BatchExposedFilter struct {
	Store ExposedStore

	Checker BatchExposureChecker

	KeyPrefix            string
	TimeWindow           int64
	BloomFilterDayWindow int
}

var _ BatchFilter = (*BatchExposedFilter)(nil)

func NewBatchExposedFilter(
	storeAdapter *StoreAdapter,
	checker BatchExposureChecker,
	keyPrefix string,
	timeWindow int64,
	bloomFilterDayWindow int,
) *BatchExposedFilter {
	var store ExposedStore
	if storeAdapter != nil {
		store = storeAdapter
	}
	if checker == nil && storeAdapter != nil {
		checker = storeAdapter
	}
	return &BatchExposedFilter{
		Store:                store,
		Checker:              checker,
		KeyPrefix:            keyPrefix,
		TimeWindow:           timeWindow,
		BloomFilterDayWindow: bloomFilterDayWindow,
	}
}

func (f *BatchExposedFilter) Name() string { return "filter.exposed_batch" }

func (f *BatchExposedFilter) ShouldFilter(ctx context.Context, rctx *core.RecommendContext, item *core.Item) (bool, error) {
	if item == nil {
		return false, nil
	}
	out, err := f.FilterBatch(ctx, rctx, []*core.Item{item})
	if err != nil {
		return false, err
	}
	return len(out) == 0, nil
}

func (f *BatchExposedFilter) FilterBatch(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if len(items) == 0 || rctx == nil || rctx.UserID == "" {
		return items, nil
	}

	keyPrefix := f.KeyPrefix
	if keyPrefix == "" {
		keyPrefix = "user:exposed"
	}

	itemIDs := make([]string, 0, len(items))
	for _, it := range items {
		if it != nil && it.ID != "" {
			itemIDs = append(itemIDs, it.ID)
		}
	}
	if len(itemIDs) == 0 {
		return items, nil
	}

	if f.Checker != nil {
		exposed, err := f.Checker.CheckExposedBatch(
			ctx, rctx.UserID, itemIDs, keyPrefix, f.TimeWindow, f.BloomFilterDayWindow,
		)
		if err == nil && len(exposed) > 0 {
			out := make([]*core.Item, 0, len(items))
			for _, it := range items {
				if it == nil {
					continue
				}
				if exposed[it.ID] {
					continue
				}
				out = append(out, it)
			}
			return out, nil
		}
	}

	// 降级：复用原有 ExposedFilter 的单条逻辑。
	legacy := &ExposedFilter{
		Store:                f.Store,
		KeyPrefix:            keyPrefix,
		TimeWindow:           f.TimeWindow,
		BloomFilterDayWindow: f.BloomFilterDayWindow,
	}
	out := make([]*core.Item, 0, len(items))
	for _, it := range items {
		if it == nil {
			continue
		}
		filtered, err := legacy.ShouldFilter(ctx, rctx, it)
		if err != nil {
			return nil, err
		}
		if !filtered {
			out = append(out, it)
		}
	}
	return out, nil
}
