package rerank

import (
	"context"
	"math/rand"
	"testing"

	"github.com/rushteam/reckit/core"
)

type mockStatsProvider struct {
	stats map[string]BanditStats
}

func (m *mockStatsProvider) BatchGetStats(_ context.Context, _ *core.RecommendContext, _ []string) (map[string]BanditStats, error) {
	return m.stats, nil
}

func TestUCBNode(t *testing.T) {
	items := []*core.Item{
		{ID: "popular", Score: 0.8},
		{ID: "cold", Score: 0.5},
		{ID: "medium", Score: 0.6},
	}
	provider := &mockStatsProvider{stats: map[string]BanditStats{
		"popular": {Impressions: 1000, Conversions: 100},
		"cold":    {Impressions: 0, Conversions: 0},
		"medium":  {Impressions: 50, Conversions: 10},
	}}
	n := &UCBNode{Provider: provider, C: 1.0}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("got %d want 3", len(out))
	}
	if out[0].ID != "cold" {
		t.Fatalf("cold item (0 impressions) should be first, got %s", out[0].ID)
	}
}

func TestThompsonSampling(t *testing.T) {
	items := []*core.Item{
		{ID: "a", Score: 0.8},
		{ID: "b", Score: 0.5},
		{ID: "c", Score: 0.3},
	}
	provider := &mockStatsProvider{stats: map[string]BanditStats{
		"a": {Impressions: 100, Conversions: 80},
		"b": {Impressions: 100, Conversions: 50},
		"c": {Impressions: 100, Conversions: 10},
	}}
	rng := rand.New(rand.NewSource(42))
	n := &ThompsonSamplingNode{Provider: provider, Rand: rng}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("got %d want 3", len(out))
	}
}

func TestThompsonSampling_NilProvider(t *testing.T) {
	items := []*core.Item{{ID: "a", Score: 0.8}}
	n := &ThompsonSamplingNode{}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 {
		t.Fatalf("got %d want 1", len(out))
	}
}

func TestBetaSample(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	sum := 0.0
	n := 10000
	for i := 0; i < n; i++ {
		sum += betaSample(rng, 10, 10)
	}
	mean := sum / float64(n)
	if mean < 0.4 || mean > 0.6 {
		t.Fatalf("Beta(10,10) mean=%.3f, expected ~0.5", mean)
	}
}
