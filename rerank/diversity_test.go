package rerank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

func makeItem(id string, score float64) *core.Item {
	it := core.NewItem(id)
	it.Score = score
	return it
}

func TestGetValue_Labels(t *testing.T) {
	d := &Diversity{}
	it := makeItem("1", 1.0)
	it.PutLabel("category", utils.Label{Value: "tech"})
	it.Meta = map[string]any{"category": "news"}
	it.Features = map[string]float64{"category": 3.0}

	if v := d.getValue(it, "category"); v != "tech" {
		t.Errorf("Labels should have highest priority: got %q, want %q", v, "tech")
	}
}

func TestGetValue_Meta(t *testing.T) {
	d := &Diversity{}
	it := makeItem("1", 1.0)
	it.Meta = map[string]any{"category": "news"}
	it.Features = map[string]float64{"category": 3.0}

	if v := d.getValue(it, "category"); v != "news" {
		t.Errorf("Meta should be second priority: got %q, want %q", v, "news")
	}
}

func TestGetValue_Features_Integer(t *testing.T) {
	d := &Diversity{}
	it := makeItem("1", 1.0)
	it.Features = map[string]float64{"category_id": 42.0}

	if v := d.getValue(it, "category_id"); v != "42" {
		t.Errorf("Features integer should format without decimal: got %q, want %q", v, "42")
	}
}

func TestGetValue_Features_Float(t *testing.T) {
	d := &Diversity{}
	it := makeItem("1", 1.0)
	it.Features = map[string]float64{"score_bucket": 0.75}

	if v := d.getValue(it, "score_bucket"); v != "0.75" {
		t.Errorf("Features float: got %q, want %q", v, "0.75")
	}
}

func TestGetValue_Missing(t *testing.T) {
	d := &Diversity{}
	it := makeItem("1", 1.0)

	if v := d.getValue(it, "missing"); v != "" {
		t.Errorf("missing key should return empty: got %q", v)
	}
}

func TestGetValue_Nil(t *testing.T) {
	d := &Diversity{}
	if v := d.getValue(nil, "key"); v != "" {
		t.Errorf("nil item should return empty: got %q", v)
	}
}

func TestDiversity_CategoryDedup_FromFeatures(t *testing.T) {
	ctx := context.Background()
	rctx := &core.RecommendContext{UserID: "u1"}

	items := []*core.Item{
		func() *core.Item {
			it := makeItem("a", 0.9)
			it.Features = map[string]float64{"category_id": 1}
			return it
		}(),
		func() *core.Item {
			it := makeItem("b", 0.8)
			it.Features = map[string]float64{"category_id": 2}
			return it
		}(),
		func() *core.Item {
			it := makeItem("c", 0.7)
			it.Features = map[string]float64{"category_id": 1}
			return it
		}(),
		func() *core.Item {
			it := makeItem("d", 0.6)
			it.Features = map[string]float64{"category_id": 3}
			return it
		}(),
	}

	d := &Diversity{LabelKey: "category_id"}
	out, err := d.Process(ctx, rctx, items)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// category_id=1 出现两次（a, c），去重后保留 a；b(2), d(3) 各一次
	if len(out) != 3 {
		t.Fatalf("want 3, got %d", len(out))
	}
	expected := []string{"a", "b", "d"}
	for i, it := range out {
		if it.ID != expected[i] {
			t.Errorf("index %d: want %s, got %s", i, expected[i], it.ID)
		}
	}
}

func TestDiversity_SingleKeyDiversity_FromFeatures(t *testing.T) {
	ctx := context.Background()
	rctx := &core.RecommendContext{UserID: "u1"}

	items := []*core.Item{
		func() *core.Item {
			it := makeItem("a", 0.9)
			it.Features = map[string]float64{"author_id": 100}
			return it
		}(),
		func() *core.Item {
			it := makeItem("b", 0.8)
			it.Features = map[string]float64{"author_id": 100}
			return it
		}(),
		func() *core.Item {
			it := makeItem("c", 0.7)
			it.Features = map[string]float64{"author_id": 200}
			return it
		}(),
	}

	d := &Diversity{DiversityKeys: []string{"author_id"}, MaxConsecutive: 1, WindowSize: 1}
	out, err := d.Process(ctx, rctx, items)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// a(100) 插入 → b(100) 延迟 → c(200) 插入，窗口滑走 100 → pending b(100) 可插入
	if len(out) != 3 {
		t.Fatalf("want 3, got %d", len(out))
	}
	if out[0].ID != "a" || out[1].ID != "c" || out[2].ID != "b" {
		t.Errorf("want [a, c, b], got [%s, %s, %s]", out[0].ID, out[1].ID, out[2].ID)
	}
}

func TestDiversity_MultiKeyDiversity_AuthorAndCategory(t *testing.T) {
	ctx := context.Background()
	rctx := &core.RecommendContext{UserID: "u1"}

	items := []*core.Item{
		func() *core.Item {
			it := makeItem("1", 0.99)
			it.Meta = map[string]any{"author": "a1", "category": "c1"}
			return it
		}(),
		func() *core.Item {
			it := makeItem("2", 0.98)
			it.Meta = map[string]any{"author": "a1", "category": "c1"}
			return it
		}(),
		func() *core.Item {
			it := makeItem("3", 0.97)
			it.Meta = map[string]any{"author": "a2", "category": "c2"}
			return it
		}(),
		func() *core.Item {
			it := makeItem("4", 0.96)
			it.Meta = map[string]any{"author": "a1", "category": "c2"}
			return it
		}(),
		func() *core.Item {
			it := makeItem("5", 0.95)
			it.Meta = map[string]any{"author": "a2", "category": "c1"}
			return it
		}(),
	}

	d := &Diversity{
		DiversityKeys:  []string{"author", "category"},
		MaxConsecutive: 1,
		WindowSize:     1,
	}
	out, err := d.Process(ctx, rctx, items)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// 多样性重排不应丢弃 item：输出数量必须等于输入数量。
	if len(out) != len(items) {
		t.Fatalf("output count %d != input count %d: items should never be dropped", len(out), len(items))
	}

	// 验证所有输入 item 都出现在输出中。
	outIDs := make(map[string]bool, len(out))
	for _, it := range out {
		outIDs[it.ID] = true
	}
	for _, it := range items {
		if !outIDs[it.ID] {
			t.Fatalf("item %s missing from output", it.ID)
		}
	}

	// 前段（被打散的部分）应尽量满足多样性约束，但尾部兜底 item 可能违反。
	// 至少验证第一对相邻 item 不会同 author。
	if len(out) >= 2 {
		a0 := d.getValue(out[0], "author")
		a1 := d.getValue(out[1], "author")
		if a0 != "" && a0 == a1 {
			t.Errorf("first two items have same author %s, diversity not working", a0)
		}
	}
}

// ---------------------------------------------------------------------------
// 高级模式：Constraints
// ---------------------------------------------------------------------------

func TestDiversity_Constraint_MaxConsecutive(t *testing.T) {
	ctx := context.Background()
	rctx := &core.RecommendContext{UserID: "u1"}

	items := []*core.Item{
		catItem("a", 0.9, "tech"),
		catItem("b", 0.8, "tech"),
		catItem("c", 0.7, "tech"),
		catItem("d", 0.6, "sports"),
		catItem("e", 0.5, "music"),
	}
	d := &Diversity{
		Constraints: []DiversityConstraint{
			{Dimensions: []string{"category"}, MaxConsecutive: 2, Weight: 1},
		},
	}
	out, err := d.Process(ctx, rctx, items)
	if err != nil {
		t.Fatal(err)
	}
	// a(tech) b(tech) → 2 consecutive OK; c(tech) would be 3 → skip, pick d(sports)
	if len(out) < 4 {
		t.Fatalf("len=%d", len(out))
	}
	if out[0].ID != "a" || out[1].ID != "b" {
		t.Fatalf("first two should be a,b; got %s,%s", out[0].ID, out[1].ID)
	}
	if out[2].ID == "c" {
		t.Fatal("c should not be third (would violate MaxConsecutive=2)")
	}
}

func TestDiversity_Constraint_WindowFrequency(t *testing.T) {
	ctx := context.Background()
	rctx := &core.RecommendContext{UserID: "u1"}

	items := []*core.Item{
		catItem("a", 0.9, "tech"),
		catItem("b", 0.8, "sports"),
		catItem("c", 0.7, "tech"),
		catItem("d", 0.6, "tech"),
		catItem("e", 0.5, "music"),
	}
	d := &Diversity{
		Constraints: []DiversityConstraint{
			{Dimensions: []string{"category"}, WindowSize: 3, MaxPerWindow: 1, Weight: 1},
		},
	}
	out, err := d.Process(ctx, rctx, items)
	if err != nil {
		t.Fatal(err)
	}
	// MaxPerWindow=1 in window=3: at most 1 tech in any 3 consecutive
	// a(tech) → b(sports) ok → c(tech): window[a,b,c] has 2 tech → skip → e(music) ok
	if len(out) < 3 {
		t.Fatalf("len=%d", len(out))
	}
	if out[2].ID == "c" {
		t.Fatal("c should not be at pos 2 (window frequency violated)")
	}
}

func TestDiversity_Constraint_WeightFallback(t *testing.T) {
	ctx := context.Background()
	rctx := &core.RecommendContext{UserID: "u1"}

	items := []*core.Item{
		catItem("a", 0.9, "tech"),
		catItem("b", 0.8, "tech"),
		catItem("c", 0.7, "tech"),
	}
	// All same category → no perfect match after first; weight fallback picks next
	d := &Diversity{
		Constraints: []DiversityConstraint{
			{Dimensions: []string{"category"}, MaxConsecutive: 1, Weight: 1},
		},
	}
	out, err := d.Process(ctx, rctx, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("len=%d want 3 (fallback should include all)", len(out))
	}
}

func TestDiversity_Constraint_MultiValue(t *testing.T) {
	ctx := context.Background()
	rctx := &core.RecommendContext{UserID: "u1"}

	mkItem := func(id string, score float64, tags string) *core.Item {
		it := core.NewItem(id)
		it.Score = score
		it.PutLabel("tags", utils.Label{Value: tags, Source: "test"})
		return it
	}

	items := []*core.Item{
		mkItem("a", 0.9, "tech|gaming"),
		mkItem("b", 0.8, "gaming|music"),  // overlaps with a on "gaming"
		mkItem("c", 0.7, "sports|health"), // no overlap with a
		mkItem("d", 0.6, "tech|sports"),   // overlaps with a on "tech"
	}
	d := &Diversity{
		Constraints: []DiversityConstraint{
			{Dimensions: []string{"tags"}, MaxConsecutive: 1, Weight: 1, MultiValueDelimiter: "|"},
		},
	}
	out, err := d.Process(ctx, rctx, items)
	if err != nil {
		t.Fatal(err)
	}
	// a(tech|gaming) → b overlaps(gaming) → skip → c(sports|health) no overlap → OK
	if len(out) < 2 {
		t.Fatalf("len=%d", len(out))
	}
	if out[1].ID != "c" {
		t.Fatalf("second should be c (no overlap); got %s", out[1].ID)
	}
}

func TestDiversity_Constraint_ExcludeChannels(t *testing.T) {
	ctx := context.Background()
	rctx := &core.RecommendContext{UserID: "u1"}

	items := []*core.Item{
		newItemWithSource("a", 0.9, "hot"),
		newItemWithSource("b", 0.8, "cf"),
		newItemWithSource("c", 0.7, "hot"),
		newItemWithSource("d", 0.6, "ann"),
	}
	for _, it := range items {
		it.PutLabel("category", utils.Label{Value: "tech", Source: "test"})
	}

	d := &Diversity{
		Constraints: []DiversityConstraint{
			{Dimensions: []string{"category"}, MaxConsecutive: 1, Weight: 1},
		},
		ExcludeChannels: []string{"hot"},
	}
	out, err := d.Process(ctx, rctx, items)
	if err != nil {
		t.Fatal(err)
	}
	// hot items excluded from diversity, appended at tail
	// candidates: b(cf), d(ann); excluded: a(hot), c(hot)
	if len(out) != 4 {
		t.Fatalf("len=%d", len(out))
	}
	// Last two should be the hot items
	if out[2].ID != "a" || out[3].ID != "c" {
		t.Fatalf("hot items should be at tail; got %s, %s", out[2].ID, out[3].ID)
	}
}
