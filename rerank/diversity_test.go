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

func TestDiversity_AuthorDiversity_FromFeatures(t *testing.T) {
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

	d := &Diversity{AuthorKey: "author_id", MaxConsecutive: 1, WindowSize: 1}
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
