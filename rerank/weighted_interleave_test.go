package rerank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestWeightedInterleave_HighWeightGetMore(t *testing.T) {
	items := []*core.Item{
		newItemWithSource("h1", 0.9, "hot"),
		newItemWithSource("h2", 0.8, "hot"),
		newItemWithSource("h3", 0.7, "hot"),
		newItemWithSource("c1", 0.6, "cf"),
		newItemWithSource("c2", 0.5, "cf"),
		newItemWithSource("c3", 0.4, "cf"),
	}
	node := &WeightedInterleaveNode{
		N:       4,
		Weights: map[string]float64{"hot": 3, "cf": 1},
	}
	out, err := node.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("len=%d", len(out))
	}
	hotCount := 0
	for _, it := range out {
		if v, _ := it.GetValue("recall_source"); v == "hot" {
			hotCount++
		}
	}
	if hotCount < 2 {
		t.Fatalf("hot=%d want >=2 (weight 3:1)", hotCount)
	}
}

func TestWeightedInterleave_EqualWeights(t *testing.T) {
	items := []*core.Item{
		newItemWithSource("a1", 0.9, "x"),
		newItemWithSource("a2", 0.8, "x"),
		newItemWithSource("b1", 0.7, "y"),
		newItemWithSource("b2", 0.6, "y"),
	}
	node := &WeightedInterleaveNode{
		N:       4,
		Weights: map[string]float64{"x": 1, "y": 1},
	}
	out, err := node.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 4 {
		t.Fatalf("len=%d", len(out))
	}
}
