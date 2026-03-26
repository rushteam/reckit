package rerank

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestEpsilonGreedy_NoExplore(t *testing.T) {
	items := scoredItems(1.0, 0.9, 0.8, 0.7, 0.6)
	n := &EpsilonGreedyNode{Epsilon: 0, ExploitSize: 3}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if out[0].ID != "0" || out[1].ID != "1" || out[2].ID != "2" {
		t.Fatal("epsilon=0 should not change order")
	}
}

func TestEpsilonGreedy_FullExplore(t *testing.T) {
	items := scoredItems(1.0, 0.9, 0.8, 0.7, 0.6)
	rng := rand.New(rand.NewSource(42))
	n := &EpsilonGreedyNode{Epsilon: 1.0, ExploitSize: 3, Rand: rng}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 5 {
		t.Fatalf("got %d want 5", len(out))
	}
	changed := false
	for i, it := range out[:3] {
		if it.ID != items[i].ID {
			changed = true
			break
		}
	}
	if !changed {
		t.Fatal("epsilon=1.0 should have changed some exploit positions")
	}
}

func TestEpsilonGreedy_TooFewItems(t *testing.T) {
	items := scoredItems(1.0, 0.9)
	n := &EpsilonGreedyNode{Epsilon: 0.5, ExploitSize: 5}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("got %d want 2", len(out))
	}
}

func scoredItems(scores ...float64) []*core.Item {
	out := make([]*core.Item, len(scores))
	for i, s := range scores {
		it := core.NewItem(string(rune('0' + i)))
		it.Score = s
		out[i] = it
	}
	return out
}
