package rerank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestColdStartBoost_Meta(t *testing.T) {
	items := []*core.Item{
		{ID: "new", Score: 0.5, Meta: map[string]any{"impressions": int64(10)}},
		{ID: "old", Score: 0.5, Meta: map[string]any{"impressions": int64(200)}},
		{ID: "zero", Score: 0.5, Meta: map[string]any{}},
	}
	n := &ColdStartBoostNode{Threshold: 100, BoostValue: 1.0}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	if out[0].Score <= 0.5 {
		t.Fatal("new item should be boosted")
	}
	if out[1].Score != 0.5 {
		t.Fatal("old item should not change")
	}
	if out[2].Score <= 0.5 {
		t.Fatal("zero impression item should be boosted")
	}
}

func TestColdStartBoost_Provider(t *testing.T) {
	items := []*core.Item{
		{ID: "cold", Score: 0.3},
		{ID: "warm", Score: 0.3},
	}
	provider := &mockStatsProvider{stats: map[string]BanditStats{
		"cold": {Impressions: 5},
		"warm": {Impressions: 500},
	}}
	n := &ColdStartBoostNode{Provider: provider, Threshold: 100, BoostValue: 2.0}
	out, err := n.Process(context.Background(), nil, items)
	if err != nil {
		t.Fatal(err)
	}
	expected := 0.3 + 2.0*(1.0-5.0/100.0)
	if diff := out[0].Score - expected; diff > 0.001 || diff < -0.001 {
		t.Fatalf("cold score=%.4f want ~%.4f", out[0].Score, expected)
	}
	if out[1].Score != 0.3 {
		t.Fatalf("warm score=%.4f want 0.3", out[1].Score)
	}
}
