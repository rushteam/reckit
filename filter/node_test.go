package filter

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

// --- test helpers ---

type staticFilter struct {
	name      string
	blockedID string
}

func (f *staticFilter) Name() string { return f.name }
func (f *staticFilter) ShouldFilter(_ context.Context, _ *core.RecommendContext, item *core.Item) (bool, error) {
	return item.ID == f.blockedID, nil
}

type batchBlockFilter struct {
	name       string
	blockedIDs map[string]bool
}

func (f *batchBlockFilter) Name() string { return f.name }
func (f *batchBlockFilter) ShouldFilter(_ context.Context, _ *core.RecommendContext, item *core.Item) (bool, error) {
	return f.blockedIDs[item.ID], nil
}
func (f *batchBlockFilter) FilterBatch(_ context.Context, _ *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	out := make([]*core.Item, 0, len(items))
	for _, it := range items {
		if it != nil && !f.blockedIDs[it.ID] {
			out = append(out, it)
		}
	}
	return out, nil
}

func makeItems(ids ...string) []*core.Item {
	items := make([]*core.Item, len(ids))
	for i, id := range ids {
		items[i] = core.NewItem(id)
	}
	return items
}

// --- tests ---

func TestFilterNode_ItemFilter(t *testing.T) {
	node := &FilterNode{
		Filters: []Filter{
			&staticFilter{name: "block_a", blockedID: "a"},
		},
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, makeItems("a", "b", "c"))
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
	if out[0].ID != "b" || out[1].ID != "c" {
		t.Errorf("want [b,c], got [%s,%s]", out[0].ID, out[1].ID)
	}
}

func TestFilterNode_BatchFilter(t *testing.T) {
	node := &FilterNode{
		Filters: []Filter{
			&batchBlockFilter{name: "batch_block", blockedIDs: map[string]bool{"a": true, "c": true}},
		},
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, makeItems("a", "b", "c", "d"))
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
	if out[0].ID != "b" || out[1].ID != "d" {
		t.Errorf("want [b,d], got [%s,%s]", out[0].ID, out[1].ID)
	}
}

func TestFilterNode_BatchThenItemFilter(t *testing.T) {
	node := &FilterNode{
		Filters: []Filter{
			&batchBlockFilter{name: "batch", blockedIDs: map[string]bool{"a": true}},
			&staticFilter{name: "item", blockedID: "c"},
		},
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, makeItems("a", "b", "c", "d"))
	if err != nil {
		t.Fatal(err)
	}
	// batch removes "a", item removes "c" → [b, d]
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
	if out[0].ID != "b" || out[1].ID != "d" {
		t.Errorf("want [b,d], got [%s,%s]", out[0].ID, out[1].ID)
	}
}

func TestFilterNode_MultipleBatchFilters(t *testing.T) {
	node := &FilterNode{
		Filters: []Filter{
			&batchBlockFilter{name: "batch1", blockedIDs: map[string]bool{"a": true}},
			&batchBlockFilter{name: "batch2", blockedIDs: map[string]bool{"c": true}},
		},
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, makeItems("a", "b", "c", "d"))
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
	if out[0].ID != "b" || out[1].ID != "d" {
		t.Errorf("want [b,d], got [%s,%s]", out[0].ID, out[1].ID)
	}
}

func TestFilterNode_EmptyFilters(t *testing.T) {
	node := &FilterNode{}
	items := makeItems("a", "b")
	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("empty filters should pass through: want 2, got %d", len(out))
	}
}

func TestFilterNode_EmptyItems(t *testing.T) {
	node := &FilterNode{
		Filters: []Filter{&staticFilter{name: "f", blockedID: "a"}},
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 0 {
		t.Fatalf("empty items should return empty: got %d", len(out))
	}
}

func TestFilterNode_FilteredLabel(t *testing.T) {
	node := &FilterNode{
		Filters: []Filter{
			&staticFilter{name: "block_a", blockedID: "a"},
		},
	}
	items := makeItems("a", "b")
	node.Process(context.Background(), &core.RecommendContext{}, items)

	lbl, ok := items[0].Labels["filtered"]
	if !ok {
		t.Fatal("filtered item should have 'filtered' label")
	}
	if lbl.Value != "true" || lbl.Source != "block_a" {
		t.Errorf("want filtered label {true, block_a}, got {%s, %s}", lbl.Value, lbl.Source)
	}
}
