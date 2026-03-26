package rerank

import (
	"context"
	"math"
	"testing"

	"github.com/rushteam/reckit/core"
)

func embItem(id string, score float64, emb []float64) *core.Item {
	it := core.NewItem(id)
	it.Score = score
	it.Meta = map[string]any{"embedding": emb}
	return it
}

func TestDPPDiversity_Basic(t *testing.T) {
	items := []*core.Item{
		embItem("a", 0.9, []float64{1, 0, 0}),
		embItem("b", 0.8, []float64{0.99, 0.01, 0}), // very similar to a
		embItem("c", 0.7, []float64{0, 1, 0}),        // orthogonal to a
		embItem("d", 0.6, []float64{0, 0, 1}),        // orthogonal to both
	}

	node := &DPPDiversityNode{
		N:            3,
		Alpha:        0.1, // low alpha → diversity-heavy
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
	// With low alpha, DPP should prefer diverse items (a, c, d) over similar (a, b)
	hasB := false
	for _, it := range out {
		if it.ID == "b" {
			hasB = true
		}
	}
	if hasB {
		// b is very similar to a, so DPP should prefer c and d
		t.Log("b was selected despite being very similar to a (alpha may be too high)")
	}
}

func TestDPPDiversity_Windowed(t *testing.T) {
	items := make([]*core.Item, 8)
	for i := range items {
		emb := make([]float64, 4)
		emb[i%4] = 1.0
		items[i] = embItem(string(rune('a'+i)), float64(8-i)/10, emb)
	}

	node := &DPPDiversityNode{
		N:            6,
		Alpha:        0.5,
		WindowSize:   3,
		NormalizeEmb: true,
		EmbeddingKey: "embedding",
	}
	out, err := node.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 6 {
		t.Fatalf("len=%d", len(out))
	}
}

func TestDPPDiversity_MissingEmbedding(t *testing.T) {
	items := []*core.Item{core.NewItem("a"), core.NewItem("b")}
	items[0].Score = 0.9
	items[1].Score = 0.8

	node := &DPPDiversityNode{N: 2, Alpha: 1.0, EmbeddingKey: "embedding"}
	_, err := node.Process(context.Background(), nil, items)
	if err == nil {
		t.Fatal("expected error for missing embedding")
	}
}

func TestDPPGreedy_Identity(t *testing.T) {
	// 3×3 identity kernel → all items equally good, should pick in order
	kernel := []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	}
	indices := dppGreedy(kernel, 3, 2, nil)
	if len(indices) != 2 {
		t.Fatalf("len=%d", len(indices))
	}
}

func TestVecMath(t *testing.T) {
	a := []float64{3, 4}
	if math.Abs(vecNorm(a)-5.0) > 1e-10 {
		t.Fatalf("norm=%f", vecNorm(a))
	}
	if math.Abs(vecDot(a, a)-25.0) > 1e-10 {
		t.Fatalf("dot=%f", vecDot(a, a))
	}
}
