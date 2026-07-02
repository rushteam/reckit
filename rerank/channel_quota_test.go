package rerank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

func makeChannelItem(id string, score float64, source string) *core.Item {
	it := core.NewItem(id)
	it.Score = score
	it.Labels["recall_source"] = utils.Label{Value: source, Source: "recall"}
	return it
}

func TestChannelQuotaMix_BasicWDRR(t *testing.T) {
	items := []*core.Item{
		makeChannelItem("h1", 10, "hot"),
		makeChannelItem("h2", 9, "hot"),
		makeChannelItem("h3", 8, "hot"),
		makeChannelItem("c1", 7, "cf"),
		makeChannelItem("c2", 6, "cf"),
		makeChannelItem("c3", 5, "cf"),
	}

	node := &ChannelQuotaMixNode{
		Channels: []ChannelQuotaSlot{
			{Key: "hot", Quota: 2},
			{Key: "cf", Quota: 3},
		},
		TopN: 5,
	}

	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 5 {
		t.Fatalf("expected 5 items, got %d", len(out))
	}

	hotCount, cfCount := 0, 0
	for _, it := range out {
		switch it.Labels["recall_source"].Value {
		case "hot":
			hotCount++
		case "cf":
			cfCount++
		}
	}
	if hotCount != 2 {
		t.Errorf("hot count = %d, want 2", hotCount)
	}
	if cfCount != 3 {
		t.Errorf("cf count = %d, want 3", cfCount)
	}
}

func TestChannelQuotaMix_ExhaustedChannelRedistribution(t *testing.T) {
	items := []*core.Item{
		makeChannelItem("h1", 10, "hot"),
		makeChannelItem("c1", 7, "cf"),
		makeChannelItem("c2", 6, "cf"),
		makeChannelItem("c3", 5, "cf"),
		makeChannelItem("c4", 4, "cf"),
	}

	node := &ChannelQuotaMixNode{
		Channels: []ChannelQuotaSlot{
			{Key: "hot", Quota: 3},
			{Key: "cf", Quota: 3},
		},
		TopN: 5,
	}

	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 5 {
		t.Fatalf("expected 5 items, got %d", len(out))
	}
}

func TestChannelQuotaMix_PassthroughUnmatched(t *testing.T) {
	items := []*core.Item{
		makeChannelItem("h1", 10, "hot"),
		makeChannelItem("u1", 5, "unknown"),
	}

	node := &ChannelQuotaMixNode{
		Channels: []ChannelQuotaSlot{
			{Key: "hot", Quota: 1},
		},
		TopN:                 1,
		PassthroughUnmatched: true,
	}

	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 items (1 hot + 1 passthrough), got %d", len(out))
	}
	if out[0].ID != "h1" {
		t.Errorf("first item should be h1, got %s", out[0].ID)
	}
	if out[1].ID != "u1" {
		t.Errorf("second item should be u1 (passthrough), got %s", out[1].ID)
	}
}

func TestChannelQuotaMix_WithDiversity(t *testing.T) {
	items := []*core.Item{
		makeChannelItem("h1", 10, "hot"),
		makeChannelItem("h2", 9, "hot"),
		makeChannelItem("h3", 8, "hot"),
	}
	items[0].Labels["category"] = utils.Label{Value: "tech", Source: "test"}
	items[1].Labels["category"] = utils.Label{Value: "tech", Source: "test"}
	items[2].Labels["category"] = utils.Label{Value: "food", Source: "test"}

	node := &ChannelQuotaMixNode{
		Channels: []ChannelQuotaSlot{
			{Key: "hot", Quota: 3},
		},
		TopN: 3,
		DiversityConstraints: []DiversityConstraint{
			{Dimensions: []string{"category"}, MaxConsecutive: 1},
		},
	}

	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// h1(tech) → h3(food) → h2(tech) (no consecutive same category)
	if len(out) < 2 {
		t.Fatalf("expected at least 2 items, got %d", len(out))
	}
	if out[0].Labels["category"].Value == out[1].Labels["category"].Value {
		t.Errorf("consecutive items should have different category")
	}
}

func TestDiversityPlacementFilter_CanPlace(t *testing.T) {
	f := NewDiversityPlacementFilter([]DiversityConstraint{
		{Dimensions: []string{"category"}, MaxConsecutive: 1},
	})

	item1 := core.NewItem("1")
	item1.Labels["category"] = utils.Label{Value: "tech", Source: "test"}
	item2 := core.NewItem("2")
	item2.Labels["category"] = utils.Label{Value: "tech", Source: "test"}
	item3 := core.NewItem("3")
	item3.Labels["category"] = utils.Label{Value: "food", Source: "test"}

	if !f.CanPlace(item1) {
		t.Error("first item should always be placeable")
	}
	f.Place(item1)

	if f.CanPlace(item2) {
		t.Error("consecutive same category should be blocked")
	}

	if !f.CanPlace(item3) {
		t.Error("different category should be placeable")
	}
	f.Place(item3)

	if !f.CanPlace(item2) {
		t.Error("after food, tech should be placeable again")
	}
}
