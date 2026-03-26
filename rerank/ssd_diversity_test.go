package rerank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestSSDDiversity_Basic(t *testing.T) {
	items := []*core.Item{
		embItem("a", 0.9, []float64{1, 0, 0}),
		embItem("b", 0.8, []float64{0.99, 0.01, 0}), // very similar to a
		embItem("c", 0.7, []float64{0, 1, 0}),        // orthogonal
		embItem("d", 0.6, []float64{0, 0, 1}),        // orthogonal
	}

	node := &SSDDiversityNode{
		N:            3,
		Gamma:        1.0, // high gamma → diversity-heavy
		WindowSize:   5,
		NormalizeEmb: true,
		EmbeddingKey: "embedding",
	}
	out, err := node.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("len=%d", len(out))
	}
	// First should be a (highest relevance); then diverse items preferred
	if out[0].ID != "a" {
		t.Fatalf("first should be a; got %s", out[0].ID)
	}
}

func TestSSDDiversity_AllSame(t *testing.T) {
	items := []*core.Item{
		embItem("a", 0.9, []float64{1, 0}),
		embItem("b", 0.8, []float64{1, 0}),
		embItem("c", 0.7, []float64{1, 0}),
	}
	node := &SSDDiversityNode{
		N:            3,
		Gamma:        0.5,
		WindowSize:   5,
		EmbeddingKey: "embedding",
	}
	out, err := node.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("len=%d", len(out))
	}
}

func TestSSDDiversity_MissingEmbedding(t *testing.T) {
	items := []*core.Item{core.NewItem("a"), core.NewItem("b")}
	items[0].Score = 0.9
	items[1].Score = 0.8

	node := &SSDDiversityNode{N: 2, Gamma: 0.5, EmbeddingKey: "embedding"}
	_, err := node.Process(context.Background(), nil, items)
	if err == nil {
		t.Fatal("expected error for missing embedding")
	}
}

func TestSSDDiversity_DiversePreferred(t *testing.T) {
	items := []*core.Item{
		embItem("a", 0.9, []float64{1, 0, 0, 0}),
		embItem("b", 0.85, []float64{0.98, 0.02, 0, 0}), // similar to a
		embItem("c", 0.7, []float64{0, 0, 1, 0}),         // orthogonal
		embItem("d", 0.6, []float64{0, 0, 0, 1}),         // orthogonal
	}

	node := &SSDDiversityNode{
		N:            3,
		Gamma:        2.0, // very high gamma
		WindowSize:   5,
		NormalizeEmb: true,
		EmbeddingKey: "embedding",
	}
	out, err := node.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("len=%d", len(out))
	}
	// With high gamma, should prefer orthogonal items c,d over similar b
	outIDs := make(map[string]bool)
	for _, it := range out {
		outIDs[it.ID] = true
	}
	if outIDs["b"] && (!outIDs["c"] || !outIDs["d"]) {
		t.Fatal("with high gamma, diverse items (c,d) should be preferred over similar b")
	}
}
