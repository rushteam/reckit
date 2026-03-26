package rerank

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestSampleNode_NoShuffle(t *testing.T) {
	items := make([]*core.Item, 10)
	for i := range items {
		items[i] = core.NewItem(string(rune('a' + i)))
	}
	node := &SampleNode{N: 3}
	out, err := node.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 || out[0].ID != "a" || out[1].ID != "b" || out[2].ID != "c" {
		t.Fatalf("got %v", ids(out))
	}
}

func TestSampleNode_Shuffle(t *testing.T) {
	items := make([]*core.Item, 20)
	for i := range items {
		items[i] = core.NewItem(string(rune('a' + i)))
	}
	node := &SampleNode{N: 5, Shuffle: true, Rand: rand.New(rand.NewSource(42))}
	out, err := node.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Fatalf("len=%d", len(out))
	}
}

func TestSampleNode_NoBiggerThanLen(t *testing.T) {
	items := []*core.Item{core.NewItem("a")}
	node := &SampleNode{N: 10, Shuffle: true}
	out, err := node.Process(context.Background(), nil, items)
	if err != nil || len(out) != 1 {
		t.Fatalf("err=%v len=%d", err, len(out))
	}
}

func ids(items []*core.Item) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = it.ID
	}
	return out
}
