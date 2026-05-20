package recall

import (
	"context"
	"fmt"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/pkg/utils"
)

// PoolItem 表示池内单个成员，通常对应 Redis ZSET 的 member + score。
type PoolItem struct {
	ID    string
	Score float64
}

// PoolReader 从外部存储读取池内全量成员。
// 典型实现：Redis ZSET ZRANGEBYSCORE / ZREVRANGEBYSCORE。
type PoolReader interface {
	// GetPoolItems 返回指定 key 的池内成员。
	// start/stop 为分数或排名范围（实现自行约定语义）；rev=true 表示降序。
	GetPoolItems(ctx context.Context, key string, start, stop int64, rev bool) ([]PoolItem, error)
}

// DistributeStrategy 定义池内 item 的分发策略接口。
// 从全量池成员中选取 TopK 分发给当前用户。
type DistributeStrategy interface {
	// Pick 从 items 中为 userID 选取 topK 条。
	Pick(ctx context.Context, req DistributeRequest) ([]PoolItem, error)
}

// DistributeRequest 一次池内分发的请求参数。
type DistributeRequest struct {
	UserID string
	Items  []PoolItem
	TopK   int
}

// NormalizeTopK 将 TopK 限制在 [0, len(Items)]。
func (r DistributeRequest) NormalizeTopK() int {
	if r.TopK <= 0 {
		return len(r.Items)
	}
	if r.TopK > len(r.Items) {
		return len(r.Items)
	}
	return r.TopK
}

// PoolRecall 从全局池读取全量成员，再经分发策略选取 TopK 给当前用户。
//
// 适用于任何"运营池子"召回场景——编辑/算法手动维护内容池，按策略均匀分发给用户。
// 是比 i2i/u2i 更基础的召回方式。
//
// 支持场景配置覆盖：设置 StrategyResolver 可在运行时动态切换分发策略。
//
// 示例：
//
//	pool := &recall.PoolRecall{
//	    Reader:     redisPoolReader,
//	    PoolKey:    "recall:editor_pick",
//	    TopK:       20,
//	    Strategy:   recall.NewUserShuffleStrategy(),
//	    SourceName: "editor_pick",
//	}
type PoolRecall struct {
	// Reader 池数据读取器（必需）
	Reader PoolReader

	// PoolKey Redis key 或其他存储标识
	PoolKey string

	// TopK 每次请求返回的最大条数，默认 20
	TopK int

	// Strategy 分发策略（必需）
	Strategy DistributeStrategy

	// SourceName 召回源标识，用于 recall_source label
	SourceName string

	// StrategyResolver 如果非 nil，每次 Recall 前调用，可动态切换分发策略。
	// 返回 nil 表示使用默认 Strategy。
	StrategyResolver func(rctx *core.RecommendContext) DistributeStrategy
}

func (r *PoolRecall) Name() string {
	if r != nil && r.SourceName != "" {
		return r.SourceName
	}
	return "recall.pool"
}

func (r *PoolRecall) Kind() pipeline.Kind { return pipeline.KindRecall }

func (r *PoolRecall) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	_ []*core.Item,
) ([]*core.Item, error) {
	return r.Recall(ctx, rctx)
}

func (r *PoolRecall) Recall(ctx context.Context, rctx *core.RecommendContext) ([]*core.Item, error) {
	if r == nil || r.Reader == nil {
		return nil, nil
	}
	if rctx == nil || rctx.UserID == "" {
		return nil, nil
	}

	all, err := r.Reader.GetPoolItems(ctx, r.PoolKey, 0, -1, true)
	if err != nil {
		return nil, fmt.Errorf("%s: read pool failed, key=%s: %w", r.Name(), r.PoolKey, err)
	}
	if len(all) == 0 {
		return nil, nil
	}

	strategy := r.Strategy
	if r.StrategyResolver != nil {
		if override := r.StrategyResolver(rctx); override != nil {
			strategy = override
		}
	}
	if strategy == nil {
		strategy = NewUserShuffleStrategy()
	}

	topK := r.TopK
	if topK <= 0 {
		topK = 20
	}

	picked, err := strategy.Pick(ctx, DistributeRequest{
		UserID: rctx.UserID,
		Items:  all,
		TopK:   topK,
	})
	if err != nil {
		return nil, fmt.Errorf("%s: distribute failed: %w", r.Name(), err)
	}

	return r.buildItems(picked), nil
}

func (r *PoolRecall) buildItems(picked []PoolItem) []*core.Item {
	out := make([]*core.Item, 0, len(picked))
	for _, p := range picked {
		if p.ID == "" {
			continue
		}
		it := core.NewItem(p.ID)
		it.Score = p.Score
		it.PutLabel("recall_source", utils.Label{Value: r.Name(), Source: "recall"})
		out = append(out, it)
	}
	return out
}
