package rerank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

func TestFairInterleave_Basic(t *testing.T) {
	a1 := newItemWithSource("a1", 0.9, "hot")
	a2 := newItemWithSource("a2", 0.5, "hot")
	b1 := newItemWithSource("b1", 0.8, "cf")
	b2 := newItemWithSource("b2", 0.3, "cf")
	c1 := newItemWithSource("c1", 0.7, "ann")

	node := &FairInterleaveNode{N: 5}
	out, err := node.Process(context.Background(), nil, []*core.Item{a1, a2, b1, b2, c1})
	if err != nil {
		t.Fatal(err)
	}
	// 轮询顺序: hot(a1), cf(b1), ann(c1), hot(a2), cf(b2)
	want := []string{"a1", "b1", "c1", "a2", "b2"}
	if len(out) != len(want) {
		t.Fatalf("len=%d want %d", len(out), len(want))
	}
	for i, w := range want {
		if out[i].ID != w {
			t.Fatalf("pos %d: got %s want %s", i, out[i].ID, w)
		}
	}
}

func TestFairInterleave_LimitN(t *testing.T) {
	items := []*core.Item{
		newItemWithSource("a1", 0.9, "hot"),
		newItemWithSource("b1", 0.8, "cf"),
		newItemWithSource("a2", 0.5, "hot"),
	}
	node := &FairInterleaveNode{N: 2}
	out, err := node.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 || out[0].ID != "a1" || out[1].ID != "b1" {
		t.Fatalf("got %v", ids(out))
	}
}

func newItemWithSource(id string, score float64, source string) *core.Item {
	it := core.NewItem(id)
	it.Score = score
	it.PutLabel("recall_source", utils.Label{Value: source, Source: "test"})
	return it
}
