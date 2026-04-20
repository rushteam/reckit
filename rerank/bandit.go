package rerank

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// BanditStats 单个物品的统计信息（曝光/转化）。
type BanditStats struct {
	Impressions int64
	Conversions int64
}

// BanditStatsProvider 批量获取物品的 bandit 统计。
type BanditStatsProvider interface {
	BatchGetStats(ctx context.Context, rctx *core.RecommendContext, itemIDs []string) (map[string]BanditStats, error)
}

// ---------------------------------------------------------------------------
// UCBNode — Upper Confidence Bound
// ---------------------------------------------------------------------------

// UCBNode 使用 UCB1 公式对物品重排：
//
//	adjusted = score + C * sqrt(ln(N) / n_i)
//
// 其中 N 为所有候选总曝光量，n_i 为当前物品曝光量，C 为探索系数。
// 曝光为 0 的物品获得 +Inf 加成（优先探索）。
type UCBNode struct {
	Provider BanditStatsProvider
	C        float64
	N        int
}

func (n *UCBNode) Name() string        { return "rerank.ucb" }
func (n *UCBNode) Kind() pipeline.Kind { return pipeline.KindReRank }

func (n *UCBNode) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.Provider == nil || len(items) <= 1 {
		return items, nil
	}

	c := n.C
	if c <= 0 {
		c = 1.0
	}

	ids := itemIDs(items)
	statsMap, err := n.Provider.BatchGetStats(ctx, rctx, ids)
	if err != nil {
		return items, nil
	}

	var totalN int64
	for _, s := range statsMap {
		totalN += s.Impressions
	}
	if totalN == 0 {
		totalN = 1
	}
	logN := math.Log(float64(totalN))

	type scored struct {
		item       *core.Item
		adjusted   float64
		coldStart  bool
	}
	ranked := make([]scored, 0, len(items))
	for _, it := range items {
		if it == nil {
			continue
		}
		s := statsMap[it.ID]
		if s.Impressions == 0 {
			ranked = append(ranked, scored{item: it, adjusted: it.Score, coldStart: true})
		} else {
			bonus := c * math.Sqrt(logN/float64(s.Impressions))
			ranked = append(ranked, scored{item: it, adjusted: it.Score + bonus, coldStart: false})
		}
	}
	// 冷启动物品（0 曝光）优先探索，组内按原 score 排序；
	// 非冷启动物品按 adjusted score 排序。
	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].coldStart != ranked[j].coldStart {
			return ranked[i].coldStart
		}
		return ranked[i].adjusted > ranked[j].adjusted
	})

	topN := n.N
	if topN <= 0 || topN > len(ranked) {
		topN = len(ranked)
	}
	out := make([]*core.Item, topN)
	for i := 0; i < topN; i++ {
		out[i] = ranked[i].item
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// ThompsonSamplingNode — Thompson Sampling (Beta-Bernoulli)
// ---------------------------------------------------------------------------

// ThompsonSamplingNode 基于 Beta 分布的汤普森采样重排。
// 对每个物品采样 θ ~ Beta(conversions+1, impressions-conversions+1)，
// 按 score * θ 排序（θ 作为乘子），或按纯 θ 排序（PureExplore=true）。
type ThompsonSamplingNode struct {
	Provider    BanditStatsProvider
	N           int
	PureExplore bool
	Rand        *rand.Rand
}

func (n *ThompsonSamplingNode) Name() string        { return "rerank.thompson_sampling" }
func (n *ThompsonSamplingNode) Kind() pipeline.Kind { return pipeline.KindReRank }

func (n *ThompsonSamplingNode) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.Provider == nil || len(items) <= 1 {
		return items, nil
	}

	rng := n.Rand
	if rng == nil {
		rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}

	ids := itemIDs(items)
	statsMap, err := n.Provider.BatchGetStats(ctx, rctx, ids)
	if err != nil {
		return items, nil
	}

	type scored struct {
		item     *core.Item
		adjusted float64
	}
	ranked := make([]scored, 0, len(items))
	for _, it := range items {
		if it == nil {
			continue
		}
		s := statsMap[it.ID]
		alpha := float64(s.Conversions + 1)
		beta := float64(s.Impressions - s.Conversions + 1)
		if beta < 1 {
			beta = 1
		}
		theta := betaSample(rng, alpha, beta)
		var adj float64
		if n.PureExplore {
			adj = theta
		} else {
			adj = it.Score * theta
		}
		ranked = append(ranked, scored{item: it, adjusted: adj})
	}
	sort.SliceStable(ranked, func(i, j int) bool { return ranked[i].adjusted > ranked[j].adjusted })

	topN := n.N
	if topN <= 0 || topN > len(ranked) {
		topN = len(ranked)
	}
	out := make([]*core.Item, topN)
	for i := 0; i < topN; i++ {
		out[i] = ranked[i].item
	}
	return out, nil
}

// betaSample 从 Beta(alpha, beta) 分布采样（使用 Gamma 分布合成）。
func betaSample(rng *rand.Rand, alpha, beta float64) float64 {
	x := gammaSample(rng, alpha)
	y := gammaSample(rng, beta)
	if x+y == 0 {
		return 0.5
	}
	return x / (x + y)
}

// gammaSample 从 Gamma(shape, 1) 分布采样；shape >= 1 用 Marsaglia-Tsang，shape < 1 用 boost。
func gammaSample(rng *rand.Rand, shape float64) float64 {
	if shape < 1 {
		u := rng.Float64()
		return gammaSample(rng, shape+1) * math.Pow(u, 1.0/shape)
	}
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)
	for {
		var x float64
		for {
			x = rng.NormFloat64()
			if 1.0+c*x > 0 {
				break
			}
		}
		v := math.Pow(1.0+c*x, 3)
		u := rng.Float64()
		if u < 1.0-0.0331*(x*x)*(x*x) {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1.0-v+math.Log(v)) {
			return d * v
		}
	}
}

func itemIDs(items []*core.Item) []string {
	ids := make([]string, 0, len(items))
	for _, it := range items {
		if it != nil {
			ids = append(ids, it.ID)
		}
	}
	return ids
}
