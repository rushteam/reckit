package recall

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/store"
)

func setupSortedSetStore(t *testing.T) *store.MemoryStore {
	t.Helper()
	ms := store.NewMemoryStore()
	ctx := context.Background()

	// hot:feed → score 降序: item_c(100) > item_b(80) > item_a(50)
	ms.ZAdd(ctx, "hot:feed", 50, "item_a")
	ms.ZAdd(ctx, "hot:feed", 80, "item_b")
	ms.ZAdd(ctx, "hot:feed", 100, "item_c")

	// price:items → score 升序场景: item_x(9.9) < item_y(29.9) < item_z(99.9)
	ms.ZAdd(ctx, "price:items", 9.9, "item_x")
	ms.ZAdd(ctx, "price:items", 29.9, "item_y")
	ms.ZAdd(ctx, "price:items", 99.9, "item_z")

	return ms
}

func TestSortedSetRecall_DescWithScores(t *testing.T) {
	ms := setupSortedSetStore(t)
	r := &SortedSetRecall{
		Store: ms, Key: "hot:feed", TopK: 10,
		Order: OrderDesc, NodeName: "recall.hot",
	}

	items, err := r.Recall(context.Background(), &core.RecommendContext{})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 3 {
		t.Fatalf("got %d items, want 3", len(items))
	}
	// desc: item_c(100) > item_b(80) > item_a(50)
	if items[0].ID != "item_c" || items[0].Score != 100 {
		t.Errorf("items[0] = %s/%.0f, want item_c/100", items[0].ID, items[0].Score)
	}
	if items[1].ID != "item_b" || items[1].Score != 80 {
		t.Errorf("items[1] = %s/%.0f, want item_b/80", items[1].ID, items[1].Score)
	}
	if items[2].ID != "item_a" || items[2].Score != 50 {
		t.Errorf("items[2] = %s/%.0f, want item_a/50", items[2].ID, items[2].Score)
	}
	// recall_source label
	if v, ok := items[0].GetValue("recall_source"); !ok || v != "recall.hot" {
		t.Errorf("recall_source = %q, want recall.hot", v)
	}
}

func TestSortedSetRecall_AscWithScores(t *testing.T) {
	ms := setupSortedSetStore(t)
	r := &SortedSetRecall{
		Store: ms, Key: "price:items", TopK: 10,
		Order: OrderAsc, NodeName: "recall.cheapest",
	}

	items, err := r.Recall(context.Background(), &core.RecommendContext{})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 3 {
		t.Fatalf("got %d items, want 3", len(items))
	}
	// asc: item_x(9.9) < item_y(29.9) < item_z(99.9)
	if items[0].ID != "item_x" {
		t.Errorf("items[0].ID = %s, want item_x", items[0].ID)
	}
	if items[2].ID != "item_z" {
		t.Errorf("items[2].ID = %s, want item_z", items[2].ID)
	}
}

func TestSortedSetRecall_TopK(t *testing.T) {
	ms := setupSortedSetStore(t)
	r := &SortedSetRecall{
		Store: ms, Key: "hot:feed", TopK: 2,
		Order: OrderDesc,
	}

	items, err := r.Recall(context.Background(), &core.RecommendContext{})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 {
		t.Fatalf("got %d items, want 2", len(items))
	}
}

func TestSortedSetRecall_KeyPrefix(t *testing.T) {
	ms := setupSortedSetStore(t)

	// KeyPrefix = "hot", Scene = "feed" → key = "hot:feed"
	r := &SortedSetRecall{
		Store: ms, KeyPrefix: "hot", TopK: 10,
		Order: OrderDesc,
	}

	rctx := &core.RecommendContext{Scene: "feed"}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 3 {
		t.Fatalf("got %d items, want 3", len(items))
	}
}

func TestSortedSetRecall_KeyPrefixNoScene(t *testing.T) {
	ms := setupSortedSetStore(t)
	ms.ZAdd(context.Background(), "trending", 10, "item_t1")

	r := &SortedSetRecall{
		Store: ms, KeyPrefix: "trending", TopK: 10,
		Order: OrderDesc,
	}

	items, err := r.Recall(context.Background(), &core.RecommendContext{})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0].ID != "item_t1" {
		t.Fatalf("unexpected items: %v", items)
	}
}

func TestSortedSetRecall_JSONFallback(t *testing.T) {
	ms := store.NewMemoryStore()
	data, _ := json.Marshal([]string{"j1", "j2", "j3"})
	ms.Set(context.Background(), "json:list", data)

	r := &SortedSetRecall{
		Store: ms, Key: "json:list", TopK: 2,
	}

	items, err := r.Recall(context.Background(), &core.RecommendContext{})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 {
		t.Fatalf("got %d items, want 2", len(items))
	}
	if items[0].ID != "j1" || items[1].ID != "j2" {
		t.Errorf("items = %s,%s, want j1,j2", items[0].ID, items[1].ID)
	}
}

func TestSortedSetRecall_StaticIDsFallback(t *testing.T) {
	r := &SortedSetRecall{
		IDs:      []string{"s1", "s2", "s3"},
		TopK:     2,
		NodeName: "recall.static",
	}

	items, err := r.Recall(context.Background(), &core.RecommendContext{})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 {
		t.Fatalf("got %d items, want 2", len(items))
	}
}

func TestSortedSetRecall_EmptyReturnsNil(t *testing.T) {
	r := &SortedSetRecall{}
	items, err := r.Recall(context.Background(), &core.RecommendContext{})
	if err != nil {
		t.Fatal(err)
	}
	if items != nil {
		t.Fatalf("expected nil, got %d items", len(items))
	}
}

func TestSortedSetRecall_DefaultName(t *testing.T) {
	r := &SortedSetRecall{}
	if r.Name() != "recall.sorted_set" {
		t.Errorf("Name() = %s, want recall.sorted_set", r.Name())
	}
}

func TestNewHotRecall(t *testing.T) {
	ms := setupSortedSetStore(t)
	r := NewHotRecall(ms, "hot:feed", 10)
	if r.Name() != "recall.hot" {
		t.Errorf("Name() = %s, want recall.hot", r.Name())
	}
	if r.Order != OrderDesc {
		t.Errorf("Order = %s, want desc", r.Order)
	}
	items, err := r.Recall(context.Background(), &core.RecommendContext{})
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 3 {
		t.Fatalf("got %d items, want 3", len(items))
	}
}

func TestNewLatestRecall(t *testing.T) {
	r := NewLatestRecall(nil, "latest:feed", 50)
	if r.Name() != "recall.latest" {
		t.Errorf("Name() = %s, want recall.latest", r.Name())
	}
	if r.Order != OrderDesc {
		t.Errorf("Order = %s, want desc", r.Order)
	}
}

func TestSortedSetRecall_Process(t *testing.T) {
	ms := setupSortedSetStore(t)
	r := NewHotRecall(ms, "hot:feed", 10)
	items, err := r.Process(context.Background(), &core.RecommendContext{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 3 {
		t.Fatalf("Process got %d items, want 3", len(items))
	}
}
