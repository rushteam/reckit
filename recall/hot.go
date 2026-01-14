package recall

import (
	"context"
	"encoding/json"
	"strconv"

	"reckit/core"
	"reckit/pipeline"
	"reckit/store"
)

// Hot 是热门召回源，支持从 Store 读取热门物品列表。
// - 如果 Store 实现了 KeyValueStore，优先使用 ZRange（有序集合，按分数排序）
// - 否则从普通 key 读取 JSON 数组
// - 如果 Store 为空，使用内存中的 IDs 作为 fallback
// Hot 同时实现了 Source 和 Node 接口，可以直接在 Pipeline 中使用
type Hot struct {
	Store store.Store
	Key   string  // 存储 key，例如 "hot:items" 或 "hot:feed"
	IDs   []int64 // fallback 内存列表
}

func (r *Hot) Name() string        { return "recall.hot" }
func (r *Hot) Kind() pipeline.Kind { return pipeline.KindRecall }

// Process 实现 Node 接口，直接调用 Recall
func (r *Hot) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	_ []*core.Item,
) ([]*core.Item, error) {
	return r.Recall(ctx, rctx)
}

// Recall 实现 Source 接口
func (r *Hot) Recall(
	ctx context.Context,
	_ *core.RecommendContext,
) ([]*core.Item, error) {
	var ids []int64

	// 优先从 Store 读取（支持 ZRange 或普通 Get）
	if r.Store != nil && r.Key != "" {
		if kvStore, ok := r.Store.(store.KeyValueStore); ok {
			// 使用有序集合：ZRange 获取 TopN（例如 Top 100）
			members, err := kvStore.ZRange(ctx, r.Key, 0, 99)
			if err == nil && len(members) > 0 {
				ids = make([]int64, 0, len(members))
				for _, m := range members {
					if id, err := strconv.ParseInt(m, 10, 64); err == nil {
						ids = append(ids, id)
					}
				}
			}
		} else {
			// 普通 key：读取 JSON 数组
			data, err := r.Store.Get(ctx, r.Key)
			if err == nil {
				var parsed []int64
				if json.Unmarshal(data, &parsed) == nil {
					ids = parsed
				}
			}
		}
	}

	// Fallback：使用内存 IDs
	if len(ids) == 0 {
		ids = r.IDs
		if len(ids) == 0 {
			ids = []int64{1, 2, 3, 4, 5} // 默认 demo 数据
		}
	}

	out := make([]*core.Item, 0, len(ids))
	for _, id := range ids {
		out = append(out, core.NewItem(id))
	}
	return out, nil
}
