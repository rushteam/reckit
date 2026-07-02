package recall

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestFanout_PreserveAlternates(t *testing.T) {
	srcA := &staticSource{name: "srcA", items: []*core.Item{
		{ID: "1", Score: 10},
		{ID: "2", Score: 8},
	}}
	srcB := &staticSource{name: "srcB", items: []*core.Item{
		{ID: "1", Score: 5},
		{ID: "3", Score: 7},
	}}

	fanout := &Fanout{
		Sources:            []Source{srcA, srcB},
		Dedup:              true,
		MergeStrategy:      &FirstMergeStrategy{},
		PreserveAlternates: true,
	}

	rctx := &core.RecommendContext{UserID: "u1"}
	items, err := fanout.Process(context.Background(), rctx, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(items) != 3 {
		t.Fatalf("expected 3 items after dedup, got %d", len(items))
	}

	// item "1" should have alternate from srcB
	item1 := findItem(items, "1")
	if item1 == nil {
		t.Fatal("item 1 not found")
	}

	altRaw, ok := item1.GetState(StateKeyAlternateSources)
	if !ok {
		t.Fatal("item 1 should have alternate sources in State")
	}

	alts, ok := altRaw.([]AlternateSource)
	if !ok {
		t.Fatalf("expected []AlternateSource, got %T", altRaw)
	}
	if len(alts) != 1 {
		t.Fatalf("expected 1 alternate, got %d", len(alts))
	}
	if alts[0].Source != "srcB" {
		t.Errorf("alternate source = %q, want srcB", alts[0].Source)
	}
	if alts[0].Score != 5 {
		t.Errorf("alternate score = %f, want 5", alts[0].Score)
	}

	// item "2" and "3" should have no alternates
	item2 := findItem(items, "2")
	if _, ok := item2.GetState(StateKeyAlternateSources); ok {
		t.Error("item 2 should not have alternates")
	}
	item3 := findItem(items, "3")
	if _, ok := item3.GetState(StateKeyAlternateSources); ok {
		t.Error("item 3 should not have alternates")
	}
}

func TestFanout_PreserveAlternates_Disabled(t *testing.T) {
	srcA := &staticSource{name: "srcA", items: []*core.Item{
		{ID: "1", Score: 10},
	}}
	srcB := &staticSource{name: "srcB", items: []*core.Item{
		{ID: "1", Score: 5},
	}}

	fanout := &Fanout{
		Sources:            []Source{srcA, srcB},
		Dedup:              true,
		MergeStrategy:      &FirstMergeStrategy{},
		PreserveAlternates: false,
	}

	rctx := &core.RecommendContext{UserID: "u1"}
	items, err := fanout.Process(context.Background(), rctx, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	if _, ok := items[0].GetState(StateKeyAlternateSources); ok {
		t.Error("alternates should not be saved when PreserveAlternates is false")
	}
}

func findItem(items []*core.Item, id string) *core.Item {
	for _, it := range items {
		if it.ID == id {
			return it
		}
	}
	return nil
}
