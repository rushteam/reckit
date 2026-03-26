package recall

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

func makeItem(id string, score float64, source string) *core.Item {
	it := core.NewItem(id)
	it.Score = score
	it.PutLabel("recall_source", utils.Label{Value: source, Source: "recall"})
	return it
}

// --- helper tests ---

func TestGroupBySource(t *testing.T) {
	items := []*core.Item{
		makeItem("1", 1.0, "hot"),
		makeItem("2", 2.0, "cf"),
		nil,
		makeItem("3", 3.0, "hot"),
	}
	groups := groupBySource(items)
	if len(groups["hot"]) != 2 {
		t.Errorf("hot group: want 2, got %d", len(groups["hot"]))
	}
	if len(groups["cf"]) != 1 {
		t.Errorf("cf group: want 1, got %d", len(groups["cf"]))
	}
}

func TestDedupItems(t *testing.T) {
	items := []*core.Item{
		makeItem("1", 1.0, "hot"),
		makeItem("1", 2.0, "cf"),
		nil,
		makeItem("2", 3.0, "hot"),
	}
	out := dedupItems(items)
	if len(out) != 2 {
		t.Fatalf("dedup: want 2, got %d", len(out))
	}
	if out[0].ID != "1" || out[0].Score != 1.0 {
		t.Errorf("dedup should keep first: got id=%s score=%f", out[0].ID, out[0].Score)
	}
	if lbl, ok := out[0].Labels["recall_source"]; !ok || lbl.Value != "hot|cf" {
		t.Errorf("dedup should merge labels: got %v", out[0].Labels["recall_source"])
	}
}

// --- WeightedScoreMergeStrategy ---

func TestWeightedScoreMerge_Basic(t *testing.T) {
	items := []*core.Item{
		makeItem("1", 0.5, "hot"),
		makeItem("2", 0.8, "cf"),
		makeItem("3", 0.3, "ann"),
	}
	s := &WeightedScoreMergeStrategy{
		SourceWeights: map[string]float64{"hot": 2.0, "cf": 1.0, "ann": 3.0},
	}
	out := s.Merge(items, false)
	if len(out) != 3 {
		t.Fatalf("want 3, got %d", len(out))
	}
	// hot: 0.5*2=1.0, cf: 0.8*1=0.8, ann: 0.3*3=0.9 -> 排序: 1.0, 0.9, 0.8
	if out[0].ID != "1" {
		t.Errorf("want item 1 first, got %s (score=%f)", out[0].ID, out[0].Score)
	}
	if out[1].ID != "3" {
		t.Errorf("want item 3 second, got %s (score=%f)", out[1].ID, out[1].Score)
	}
}

func TestWeightedScoreMerge_TopN(t *testing.T) {
	items := []*core.Item{
		makeItem("1", 0.5, "hot"),
		makeItem("2", 0.8, "cf"),
		makeItem("3", 0.3, "ann"),
	}
	s := &WeightedScoreMergeStrategy{
		SourceWeights: map[string]float64{"hot": 2.0, "cf": 1.0, "ann": 3.0},
		TopN:          2,
	}
	out := s.Merge(items, false)
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
}

func TestWeightedScoreMerge_DefaultWeight(t *testing.T) {
	items := []*core.Item{
		makeItem("1", 0.5, "hot"),
		makeItem("2", 0.8, "unknown"),
	}
	s := &WeightedScoreMergeStrategy{
		SourceWeights: map[string]float64{"hot": 2.0},
		DefaultWeight: 0.5,
	}
	out := s.Merge(items, false)
	// hot: 0.5*2=1.0, unknown: 0.8*0.5=0.4
	if out[0].ID != "1" || out[0].Score != 1.0 {
		t.Errorf("hot item: want score 1.0, got %f", out[0].Score)
	}
	if out[1].Score != 0.4 {
		t.Errorf("unknown item: want score 0.4, got %f", out[1].Score)
	}
}

func TestWeightedScoreMerge_Dedup(t *testing.T) {
	items := []*core.Item{
		makeItem("1", 0.5, "hot"),
		makeItem("1", 0.8, "cf"),
	}
	s := &WeightedScoreMergeStrategy{
		SourceWeights: map[string]float64{"hot": 2.0, "cf": 1.0},
	}
	out := s.Merge(items, true)
	if len(out) != 1 {
		t.Fatalf("dedup: want 1, got %d", len(out))
	}
}

// --- QuotaMergeStrategy ---

func TestQuotaMerge_Basic(t *testing.T) {
	items := []*core.Item{
		makeItem("1", 0.9, "hot"),
		makeItem("2", 0.7, "hot"),
		makeItem("3", 0.5, "hot"),
		makeItem("4", 0.8, "cf"),
		makeItem("5", 0.6, "cf"),
	}
	s := &QuotaMergeStrategy{
		SourceQuotas: map[string]int{"hot": 2, "cf": 1},
	}
	out := s.Merge(items, false)
	if len(out) != 3 {
		t.Fatalf("want 3, got %d", len(out))
	}
	hotCount, cfCount := 0, 0
	for _, it := range out {
		if lbl, ok := it.Labels["recall_source"]; ok {
			switch lbl.Value {
			case "hot":
				hotCount++
			case "cf":
				cfCount++
			}
		}
	}
	if hotCount != 2 {
		t.Errorf("hot: want 2, got %d", hotCount)
	}
	if cfCount != 1 {
		t.Errorf("cf: want 1, got %d", cfCount)
	}
}

func TestQuotaMerge_DefaultQuota(t *testing.T) {
	items := []*core.Item{
		makeItem("1", 0.9, "hot"),
		makeItem("2", 0.7, "hot"),
		makeItem("3", 0.8, "unknown"),
	}
	s := &QuotaMergeStrategy{
		SourceQuotas: map[string]int{"hot": 1},
		DefaultQuota: 0,
	}
	out := s.Merge(items, false)
	if len(out) != 1 {
		t.Fatalf("want 1, got %d", len(out))
	}
	if out[0].ID != "1" {
		t.Errorf("want highest score hot item, got %s", out[0].ID)
	}
}

func TestQuotaMerge_InsufficientItems(t *testing.T) {
	items := []*core.Item{
		makeItem("1", 0.9, "hot"),
	}
	s := &QuotaMergeStrategy{
		SourceQuotas: map[string]int{"hot": 10},
	}
	out := s.Merge(items, false)
	if len(out) != 1 {
		t.Fatalf("want 1 (all available), got %d", len(out))
	}
}

// --- RatioMergeStrategy ---

func TestRatioMerge_Basic(t *testing.T) {
	items := make([]*core.Item, 0, 30)
	for i := 0; i < 10; i++ {
		items = append(items, makeItem("hot_"+string(rune('a'+i)), float64(10-i), "hot"))
	}
	for i := 0; i < 10; i++ {
		items = append(items, makeItem("cf_"+string(rune('a'+i)), float64(10-i), "cf"))
	}
	for i := 0; i < 10; i++ {
		items = append(items, makeItem("ann_"+string(rune('a'+i)), float64(10-i), "ann"))
	}

	s := &RatioMergeStrategy{
		SourceRatios: map[string]float64{"hot": 0.2, "cf": 0.3, "ann": 0.5},
		TotalLimit:   10,
	}
	out := s.Merge(items, false)
	if len(out) != 10 {
		t.Fatalf("want 10, got %d", len(out))
	}

	counts := make(map[string]int)
	for _, it := range out {
		if lbl, ok := it.Labels["recall_source"]; ok {
			counts[lbl.Value]++
		}
	}
	if counts["hot"] != 2 {
		t.Errorf("hot: want 2, got %d", counts["hot"])
	}
	if counts["cf"] != 3 {
		t.Errorf("cf: want 3, got %d", counts["cf"])
	}
	if counts["ann"] != 5 {
		t.Errorf("ann: want 5, got %d", counts["ann"])
	}
}

func TestRatioMerge_Redistribution(t *testing.T) {
	items := []*core.Item{
		makeItem("hot_1", 1.0, "hot"),
		makeItem("cf_1", 5.0, "cf"),
		makeItem("cf_2", 4.0, "cf"),
		makeItem("cf_3", 3.0, "cf"),
		makeItem("cf_4", 2.0, "cf"),
		makeItem("cf_5", 1.0, "cf"),
	}
	s := &RatioMergeStrategy{
		SourceRatios: map[string]float64{"hot": 0.5, "cf": 0.5},
		TotalLimit:   4,
	}
	out := s.Merge(items, false)
	// hot 配额 2 但只有 1 个，cf 配额 2 + 余量补 1 = 3
	if len(out) != 4 {
		t.Fatalf("want 4, got %d", len(out))
	}
}

func TestRatioMerge_ZeroTotalLimit(t *testing.T) {
	items := []*core.Item{makeItem("1", 1.0, "hot")}
	s := &RatioMergeStrategy{TotalLimit: 0}
	out := s.Merge(items, false)
	if out != nil {
		t.Errorf("want nil, got %v", out)
	}
}

func TestRatioMerge_Normalization(t *testing.T) {
	items := make([]*core.Item, 0, 20)
	for i := 0; i < 10; i++ {
		items = append(items, makeItem("a_"+string(rune('a'+i)), float64(10-i), "a"))
	}
	for i := 0; i < 10; i++ {
		items = append(items, makeItem("b_"+string(rune('a'+i)), float64(10-i), "b"))
	}
	// 比例 1:3 (总和 4)，TotalLimit=8 -> a: 2, b: 6
	s := &RatioMergeStrategy{
		SourceRatios: map[string]float64{"a": 1.0, "b": 3.0},
		TotalLimit:   8,
	}
	out := s.Merge(items, false)
	if len(out) != 8 {
		t.Fatalf("want 8, got %d", len(out))
	}
	counts := make(map[string]int)
	for _, it := range out {
		if lbl, ok := it.Labels["recall_source"]; ok {
			counts[lbl.Value]++
		}
	}
	if counts["a"] != 2 {
		t.Errorf("a: want 2, got %d", counts["a"])
	}
	if counts["b"] != 6 {
		t.Errorf("b: want 6, got %d", counts["b"])
	}
}

func TestHybridRatioMerge_KeepUnconfiguredAndAllocateExplicit(t *testing.T) {
	items := []*core.Item{
		makeItem("u1", 9, "unconfigured"),
		makeItem("u2", 8, "unconfigured"),
		makeItem("a1", 7, "a"),
		makeItem("a2", 6, "a"),
		makeItem("b1", 7, "b"),
		makeItem("b2", 6, "b"),
	}
	s := &HybridRatioMergeStrategy{
		SourceRatios: map[string]float64{"a": 1, "b": 1},
		TotalLimit:   5,
	}
	out := s.Merge(items, true)
	if len(out) != 5 {
		t.Fatalf("want 5, got %d", len(out))
	}

	counts := map[string]int{}
	for _, it := range out {
		counts[it.Labels["recall_source"].Value]++
	}
	// 未配置源保留 2 个，剩余 3 个按 1:1 分配（2 + 1）
	if counts["unconfigured"] != 2 {
		t.Fatalf("unconfigured want 2, got %d", counts["unconfigured"])
	}
	if counts["a"]+counts["b"] != 3 {
		t.Fatalf("configured total want 3, got %d", counts["a"]+counts["b"])
	}
}

func TestHybridRatioMerge_DropUnconfigured(t *testing.T) {
	items := []*core.Item{
		makeItem("u1", 9, "unconfigured"),
		makeItem("a1", 7, "a"),
		makeItem("b1", 7, "b"),
	}
	s := &HybridRatioMergeStrategy{
		SourceRatios:            map[string]float64{"a": 1, "b": 1},
		TotalLimit:              2,
		DropUnconfiguredSources: true,
	}
	out := s.Merge(items, true)
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
	for _, it := range out {
		if it.Labels["recall_source"].Value == "unconfigured" {
			t.Fatal("unconfigured source should be dropped")
		}
	}
}

// --- RoundRobinMergeStrategy ---

func TestRoundRobinMerge_Basic(t *testing.T) {
	items := []*core.Item{
		makeItem("h1", 3.0, "hot"),
		makeItem("h2", 2.0, "hot"),
		makeItem("h3", 1.0, "hot"),
		makeItem("c1", 3.0, "cf"),
		makeItem("c2", 2.0, "cf"),
		makeItem("a1", 3.0, "ann"),
	}
	s := &RoundRobinMergeStrategy{
		SourceOrder: []string{"hot", "cf", "ann"},
	}
	out := s.Merge(items, false)
	// 期望顺序: h1, c1, a1, h2, c2, h3
	expected := []string{"h1", "c1", "a1", "h2", "c2", "h3"}
	if len(out) != len(expected) {
		t.Fatalf("want %d, got %d", len(expected), len(out))
	}
	for i, it := range out {
		if it.ID != expected[i] {
			t.Errorf("index %d: want %s, got %s", i, expected[i], it.ID)
		}
	}
}

func TestRoundRobinMerge_TopN(t *testing.T) {
	items := []*core.Item{
		makeItem("h1", 3.0, "hot"),
		makeItem("h2", 2.0, "hot"),
		makeItem("c1", 3.0, "cf"),
		makeItem("c2", 2.0, "cf"),
	}
	s := &RoundRobinMergeStrategy{
		SourceOrder: []string{"hot", "cf"},
		TopN:        3,
	}
	out := s.Merge(items, false)
	if len(out) != 3 {
		t.Fatalf("want 3, got %d", len(out))
	}
	expected := []string{"h1", "c1", "h2"}
	for i, it := range out {
		if it.ID != expected[i] {
			t.Errorf("index %d: want %s, got %s", i, expected[i], it.ID)
		}
	}
}

func TestRoundRobinMerge_AutoOrder(t *testing.T) {
	items := []*core.Item{
		makeItem("h1", 3.0, "hot"),
		makeItem("c1", 3.0, "cf"),
		makeItem("h2", 2.0, "hot"),
	}
	s := &RoundRobinMergeStrategy{}
	out := s.Merge(items, false)
	if len(out) != 3 {
		t.Fatalf("want 3, got %d", len(out))
	}
	// 自动检测顺序: hot(首次出现), cf
	if out[0].ID != "h1" {
		t.Errorf("first: want h1, got %s", out[0].ID)
	}
	if out[1].ID != "c1" {
		t.Errorf("second: want c1, got %s", out[1].ID)
	}
}

// --- WaterfallMergeStrategy ---

func TestWaterfallMerge_Basic(t *testing.T) {
	items := []*core.Item{
		makeItem("c1", 3.0, "cf"),
		makeItem("c2", 2.0, "cf"),
		makeItem("h1", 3.0, "hot"),
		makeItem("h2", 2.0, "hot"),
		makeItem("h3", 1.0, "hot"),
	}
	s := &WaterfallMergeStrategy{
		SourcePriority: []string{"cf", "hot"},
		TotalLimit:     4,
	}
	out := s.Merge(items, false)
	if len(out) != 4 {
		t.Fatalf("want 4, got %d", len(out))
	}
	// cf 有 2 个全取，再从 hot 取 2 个补满
	cfCount, hotCount := 0, 0
	for _, it := range out {
		if lbl, ok := it.Labels["recall_source"]; ok {
			switch lbl.Value {
			case "cf":
				cfCount++
			case "hot":
				hotCount++
			}
		}
	}
	if cfCount != 2 {
		t.Errorf("cf: want 2, got %d", cfCount)
	}
	if hotCount != 2 {
		t.Errorf("hot: want 2, got %d", hotCount)
	}
}

func TestWaterfallMerge_WithSourceLimits(t *testing.T) {
	items := []*core.Item{
		makeItem("c1", 5.0, "cf"),
		makeItem("c2", 4.0, "cf"),
		makeItem("c3", 3.0, "cf"),
		makeItem("c4", 2.0, "cf"),
		makeItem("c5", 1.0, "cf"),
		makeItem("h1", 3.0, "hot"),
		makeItem("h2", 2.0, "hot"),
		makeItem("h3", 1.0, "hot"),
	}
	s := &WaterfallMergeStrategy{
		SourcePriority: []string{"cf", "hot"},
		TotalLimit:     5,
		SourceLimits:   map[string]int{"cf": 3},
	}
	out := s.Merge(items, false)
	if len(out) != 5 {
		t.Fatalf("want 5, got %d", len(out))
	}
	cfCount, hotCount := 0, 0
	for _, it := range out {
		if lbl, ok := it.Labels["recall_source"]; ok {
			switch lbl.Value {
			case "cf":
				cfCount++
			case "hot":
				hotCount++
			}
		}
	}
	// cf 限制 3 个，hot 补 2 个
	if cfCount != 3 {
		t.Errorf("cf: want 3, got %d", cfCount)
	}
	if hotCount != 2 {
		t.Errorf("hot: want 2, got %d", hotCount)
	}
}

func TestWaterfallMerge_ZeroTotalLimit(t *testing.T) {
	items := []*core.Item{makeItem("1", 1.0, "hot")}
	s := &WaterfallMergeStrategy{TotalLimit: 0}
	out := s.Merge(items, false)
	if out != nil {
		t.Errorf("want nil, got %v", out)
	}
}

func TestWaterfallMerge_UnlistedSources(t *testing.T) {
	items := []*core.Item{
		makeItem("c1", 3.0, "cf"),
		makeItem("x1", 5.0, "extra"),
		makeItem("h1", 3.0, "hot"),
	}
	s := &WaterfallMergeStrategy{
		SourcePriority: []string{"cf"},
		TotalLimit:     3,
	}
	out := s.Merge(items, false)
	if len(out) != 3 {
		t.Fatalf("want 3, got %d", len(out))
	}
	// cf 优先，然后 extra 和 hot 按出现顺序补充
	if out[0].ID != "c1" {
		t.Errorf("first: want c1 (prioritized), got %s", out[0].ID)
	}
}

func TestWaterfallMerge_Dedup(t *testing.T) {
	items := []*core.Item{
		makeItem("1", 3.0, "cf"),
		makeItem("1", 5.0, "hot"),
		makeItem("2", 2.0, "hot"),
	}
	s := &WaterfallMergeStrategy{
		SourcePriority: []string{"cf", "hot"},
		TotalLimit:     2,
	}
	out := s.Merge(items, true)
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
}

// --- ChainMergeStrategy ---

func TestChainMerge_WeightedThenQuota(t *testing.T) {
	items := []*core.Item{
		makeItem("h1", 0.5, "hot"),
		makeItem("h2", 0.3, "hot"),
		makeItem("c1", 0.8, "cf"),
		makeItem("c2", 0.6, "cf"),
		makeItem("c3", 0.4, "cf"),
	}
	s := &ChainMergeStrategy{
		Strategies: []MergeStrategy{
			&WeightedScoreMergeStrategy{
				SourceWeights: map[string]float64{"hot": 2.0, "cf": 1.0},
			},
			&QuotaMergeStrategy{
				SourceQuotas: map[string]int{"hot": 1, "cf": 2},
			},
		},
	}
	out := s.Merge(items, true)
	if len(out) != 3 {
		t.Fatalf("want 3, got %d", len(out))
	}
	// hot: h1=1.0, h2=0.6 -> quota 1 -> h1
	// cf: c1=0.8, c2=0.6, c3=0.4 -> quota 2 -> c1, c2
	hotCount, cfCount := 0, 0
	for _, it := range out {
		if lbl, ok := it.Labels["recall_source"]; ok {
			switch lbl.Value {
			case "hot":
				hotCount++
			case "cf":
				cfCount++
			}
		}
	}
	if hotCount != 1 {
		t.Errorf("hot: want 1, got %d", hotCount)
	}
	if cfCount != 2 {
		t.Errorf("cf: want 2, got %d", cfCount)
	}
}

func TestChainMerge_RatioThenRoundRobin(t *testing.T) {
	items := make([]*core.Item, 0, 20)
	for i := 0; i < 10; i++ {
		items = append(items, makeItem("h"+string(rune('a'+i)), float64(10-i), "hot"))
	}
	for i := 0; i < 10; i++ {
		items = append(items, makeItem("c"+string(rune('a'+i)), float64(10-i), "cf"))
	}
	s := &ChainMergeStrategy{
		Strategies: []MergeStrategy{
			&RatioMergeStrategy{
				SourceRatios: map[string]float64{"hot": 0.5, "cf": 0.5},
				TotalLimit:   6,
			},
			&RoundRobinMergeStrategy{
				SourceOrder: []string{"hot", "cf"},
			},
		},
	}
	out := s.Merge(items, false)
	// Ratio 取 hot:3, cf:3 -> RoundRobin 交叉: h, c, h, c, h, c
	if len(out) != 6 {
		t.Fatalf("want 6, got %d", len(out))
	}
	for i, it := range out {
		expected := "hot"
		if i%2 == 1 {
			expected = "cf"
		}
		if lbl, ok := it.Labels["recall_source"]; ok {
			if lbl.Value != expected {
				t.Errorf("index %d: want source %s, got %s", i, expected, lbl.Value)
			}
		}
	}
}

func TestChainMerge_Empty(t *testing.T) {
	items := []*core.Item{makeItem("1", 1.0, "hot")}
	s := &ChainMergeStrategy{}
	out := s.Merge(items, false)
	if len(out) != 1 {
		t.Errorf("empty chain should pass through: want 1, got %d", len(out))
	}
}

func TestChainMerge_DedupOnlyFirst(t *testing.T) {
	items := []*core.Item{
		makeItem("1", 3.0, "hot"),
		makeItem("1", 5.0, "cf"),
		makeItem("2", 2.0, "hot"),
	}
	s := &ChainMergeStrategy{
		Strategies: []MergeStrategy{
			&WeightedScoreMergeStrategy{
				SourceWeights: map[string]float64{"hot": 1.0, "cf": 1.0},
			},
		},
	}
	out := s.Merge(items, true)
	if len(out) != 2 {
		t.Fatalf("dedup should apply in first strategy: want 2, got %d", len(out))
	}
}

// --- Fanout nesting (Fanout as Source) ---

func TestFanout_NestedAsSource(t *testing.T) {
	ctx := context.Background()
	rctx := &core.RecommendContext{UserID: "u1"}

	personalizedFanout := &Fanout{
		NodeName: "recall.personalized",
		Sources: []Source{
			&Hot{IDs: []string{"p1", "p2", "p3"}},
		},
		Dedup:         true,
		MergeStrategy: &FirstMergeStrategy{},
	}

	nonPersonalizedFanout := &Fanout{
		NodeName: "recall.non_personalized",
		Sources: []Source{
			&Hot{IDs: []string{"n1", "n2"}},
		},
		Dedup:         true,
		MergeStrategy: &FirstMergeStrategy{},
	}

	topFanout := &Fanout{
		NodeName: "recall.top",
		Sources: []Source{
			personalizedFanout,
			nonPersonalizedFanout,
		},
		Dedup:         true,
		MergeStrategy: &FirstMergeStrategy{},
	}

	items, err := topFanout.Process(ctx, rctx, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 5 {
		t.Fatalf("want 5 items, got %d", len(items))
	}

	if topFanout.Name() != "recall.top" {
		t.Errorf("top fanout name: want recall.top, got %s", topFanout.Name())
	}
	if personalizedFanout.Name() != "recall.personalized" {
		t.Errorf("personalized fanout name: want recall.personalized, got %s", personalizedFanout.Name())
	}
}

func TestFanout_NestedWithDifferentStrategies(t *testing.T) {
	ctx := context.Background()
	rctx := &core.RecommendContext{UserID: "u1"}

	cfSource := &Hot{IDs: []string{"c1", "c2", "c3", "c4", "c5"}}
	annSource := &Hot{IDs: []string{"a1", "a2", "a3"}}
	hotSource := &Hot{IDs: []string{"h1", "h2", "h3", "h4"}}

	personalizedFanout := &Fanout{
		NodeName: "recall.personalized",
		Sources:  []Source{cfSource, annSource},
		Dedup:    true,
		MergeStrategy: &WaterfallMergeStrategy{
			SourcePriority: []string{"recall.hot", "recall.hot"},
			TotalLimit:     6,
		},
	}

	topFanout := &Fanout{
		NodeName: "recall.top",
		Sources: []Source{
			personalizedFanout,
			hotSource,
		},
		Dedup: true,
		MergeStrategy: &QuotaMergeStrategy{
			SourceQuotas: map[string]int{
				"recall.personalized": 4,
				"recall.hot":          2,
			},
			DefaultQuota: 10,
		},
	}

	items, err := topFanout.Process(ctx, rctx, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) == 0 {
		t.Fatal("expected non-empty results from nested fanout")
	}
}

func TestFanout_DefaultName(t *testing.T) {
	f := &Fanout{}
	if f.Name() != "recall.fanout" {
		t.Errorf("default name: want recall.fanout, got %s", f.Name())
	}
}

// --- PriorityMergeStrategy ---

func makeItemWithPriority(id string, score float64, source string, priority int) *core.Item {
	it := makeItem(id, score, source)
	it.PutLabel("recall_priority", utils.Label{
		Value:  string(rune('0' + priority)),
		Source: "recall",
	})
	return it
}

func TestPriorityMerge_Deterministic(t *testing.T) {
	buildItems := func() []*core.Item {
		return []*core.Item{
			makeItemWithPriority("a", 1.0, "hot", 0),
			makeItemWithPriority("b", 1.0, "cf", 1),
			makeItemWithPriority("c", 1.0, "ann", 2),
			makeItemWithPriority("d", 1.0, "hot", 0),
			makeItemWithPriority("e", 1.0, "cf", 1),
		}
	}
	s := &PriorityMergeStrategy{}

	first := s.Merge(buildItems(), true)
	for i := 0; i < 100; i++ {
		out := s.Merge(buildItems(), true)
		if len(out) != len(first) {
			t.Fatalf("run %d: length changed: %d vs %d", i, len(first), len(out))
		}
		for j := range out {
			if out[j].ID != first[j].ID {
				t.Fatalf("run %d, index %d: got %s, want %s", i, j, out[j].ID, first[j].ID)
			}
		}
	}
}

func TestPriorityMerge_OrderByPriorityThenID(t *testing.T) {
	items := []*core.Item{
		makeItemWithPriority("z", 1.0, "ann", 2),
		makeItemWithPriority("a", 1.0, "hot", 0),
		makeItemWithPriority("m", 1.0, "cf", 1),
		makeItemWithPriority("b", 1.0, "hot", 0),
	}
	s := &PriorityMergeStrategy{}
	out := s.Merge(items, true)

	expected := []string{"a", "b", "m", "z"}
	if len(out) != len(expected) {
		t.Fatalf("want %d, got %d", len(expected), len(out))
	}
	for i, it := range out {
		if it.ID != expected[i] {
			t.Errorf("index %d: want %s, got %s", i, expected[i], it.ID)
		}
	}
}

func TestPriorityMerge_DedupKeepsHigherPriority(t *testing.T) {
	items := []*core.Item{
		makeItemWithPriority("1", 0.5, "cf", 2),
		makeItemWithPriority("1", 0.9, "hot", 0),
	}
	s := &PriorityMergeStrategy{}
	out := s.Merge(items, true)
	if len(out) != 1 {
		t.Fatalf("want 1, got %d", len(out))
	}
	if lbl, ok := out[0].Labels["recall_source"]; !ok || lbl.Value != "hot" {
		t.Errorf("should keep hot (priority 0), got %v", out[0].Labels["recall_source"])
	}
}

func TestPriorityMerge_NoDedupPassThrough(t *testing.T) {
	items := []*core.Item{
		makeItemWithPriority("1", 0.5, "hot", 0),
		makeItemWithPriority("1", 0.9, "cf", 1),
	}
	s := &PriorityMergeStrategy{}
	out := s.Merge(items, false)
	if len(out) != 2 {
		t.Fatalf("no dedup: want 2, got %d", len(out))
	}
}

func TestPriorityMerge_CustomWeights(t *testing.T) {
	items := []*core.Item{
		makeItem("a", 1.0, "cf"),
		makeItem("b", 1.0, "hot"),
	}
	s := &PriorityMergeStrategy{
		PriorityWeights: map[string]int{"hot": 0, "cf": 1},
	}
	out := s.Merge(items, true)
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
	if out[0].ID != "b" {
		t.Errorf("hot (weight 0) should be first, got %s", out[0].ID)
	}
}

// --- Determinism tests for Quota and Ratio ---

func TestQuotaMerge_Deterministic(t *testing.T) {
	buildItems := func() []*core.Item {
		return []*core.Item{
			makeItem("h1", 0.9, "hot"),
			makeItem("h2", 0.7, "hot"),
			makeItem("c1", 0.8, "cf"),
			makeItem("c2", 0.6, "cf"),
			makeItem("a1", 0.5, "ann"),
		}
	}
	s := &QuotaMergeStrategy{
		SourceQuotas: map[string]int{"hot": 1, "cf": 1, "ann": 1},
	}
	first := s.Merge(buildItems(), false)
	for i := 0; i < 100; i++ {
		out := s.Merge(buildItems(), false)
		if len(out) != len(first) {
			t.Fatalf("run %d: length changed", i)
		}
		for j := range out {
			if out[j].ID != first[j].ID {
				t.Fatalf("run %d, index %d: got %s, want %s", i, j, out[j].ID, first[j].ID)
			}
		}
	}
}

func TestRatioMerge_Deterministic(t *testing.T) {
	buildItems := func() []*core.Item {
		var items []*core.Item
		for i := 0; i < 5; i++ {
			items = append(items, makeItem("h"+string(rune('a'+i)), float64(5-i), "hot"))
		}
		for i := 0; i < 5; i++ {
			items = append(items, makeItem("c"+string(rune('a'+i)), float64(5-i), "cf"))
		}
		return items
	}
	s := &RatioMergeStrategy{
		SourceRatios: map[string]float64{"hot": 0.4, "cf": 0.6},
		TotalLimit:   5,
	}
	first := s.Merge(buildItems(), false)
	for i := 0; i < 100; i++ {
		out := s.Merge(buildItems(), false)
		if len(out) != len(first) {
			t.Fatalf("run %d: length changed", i)
		}
		for j := range out {
			if out[j].ID != first[j].ID {
				t.Fatalf("run %d, index %d: got %s, want %s", i, j, out[j].ID, first[j].ID)
			}
		}
	}
}
