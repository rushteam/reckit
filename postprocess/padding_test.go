package postprocess

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestPaddingNode_NoNeed(t *testing.T) {
	items := makeItems("a", "b", "c")
	n := &PaddingNode{N: 3}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("got %d want 3", len(out))
	}
}

func TestPaddingNode_Static(t *testing.T) {
	items := makeItems("a")
	fb := makeItems("b", "c", "d")
	n := &PaddingNode{N: 3, FallbackItems: fb}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("got %d want 3", len(out))
	}
	if out[0].ID != "a" || out[1].ID != "b" || out[2].ID != "c" {
		t.Fatalf("unexpected order: %s %s %s", out[0].ID, out[1].ID, out[2].ID)
	}
}

func TestPaddingNode_Dedup(t *testing.T) {
	items := makeItems("a", "b")
	fb := makeItems("a", "c", "d")
	n := &PaddingNode{N: 4, FallbackItems: fb}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("got %d want 4", len(out))
	}
	ids := map[string]bool{}
	for _, it := range out {
		if ids[it.ID] {
			t.Fatalf("duplicate %s", it.ID)
		}
		ids[it.ID] = true
	}
}

func TestPaddingNode_FallbackFunc(t *testing.T) {
	items := makeItems("a")
	n := &PaddingNode{
		N: 3,
		FallbackFunc: func(_ context.Context, _ *core.RecommendContext, need int) ([]*core.Item, error) {
			return makeItems("x", "y"), nil
		},
	}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("got %d want 3", len(out))
	}
}

func TestPaddingNode_Label(t *testing.T) {
	items := makeItems("a")
	n := &PaddingNode{N: 2, FallbackItems: makeItems("b")}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := out[0].Labels["__padding__"]; ok {
		t.Fatal("original item should not have padding label")
	}
	if _, ok := out[1].Labels["__padding__"]; !ok {
		t.Fatal("padding item missing label")
	}
}

func makeItems(ids ...string) []*core.Item {
	out := make([]*core.Item, len(ids))
	for i, id := range ids {
		out[i] = core.NewItem(id)
	}
	return out
}
