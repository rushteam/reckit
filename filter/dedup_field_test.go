package filter

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestDedupByField(t *testing.T) {
	items := []*core.Item{
		withMeta(core.NewItem("1"), "article_id", "A"),
		withMeta(core.NewItem("2"), "article_id", "B"),
		withMeta(core.NewItem("3"), "article_id", "A"),
		withMeta(core.NewItem("4"), "article_id", "C"),
	}
	f := &DedupByFieldFilter{FieldKey: "article_id"}
	out, err := f.FilterBatch(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("got %d want 3", len(out))
	}
	if out[0].ID != "1" || out[1].ID != "2" || out[2].ID != "4" {
		t.Fatalf("unexpected: %s %s %s", out[0].ID, out[1].ID, out[2].ID)
	}
}

func TestDedupByField_NoField(t *testing.T) {
	items := []*core.Item{core.NewItem("1"), core.NewItem("2")}
	f := &DedupByFieldFilter{FieldKey: "missing"}
	out, err := f.FilterBatch(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("got %d want 2", len(out))
	}
}

func withMeta(it *core.Item, k string, v any) *core.Item {
	if it.Meta == nil {
		it.Meta = map[string]any{}
	}
	it.Meta[k] = v
	return it
}
