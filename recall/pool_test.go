package recall

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

type fakePoolReader struct {
	items []PoolItem
	err   error
}

func (r *fakePoolReader) GetPoolItems(_ context.Context, _ string, _, _ int64, _ bool) ([]PoolItem, error) {
	return r.items, r.err
}

func TestPoolRecall_Basic(t *testing.T) {
	reader := &fakePoolReader{
		items: []PoolItem{
			{ID: "a", Score: 1.0},
			{ID: "b", Score: 2.0},
			{ID: "c", Score: 3.0},
		},
	}

	pool := &PoolRecall{
		Reader:     reader,
		PoolKey:    "pool:test",
		TopK:       2,
		Strategy:   NewUserShuffleStrategy(),
		SourceName: "test_pool",
	}

	rctx := &core.RecommendContext{UserID: "user-1", Params: map[string]any{}}
	out, err := pool.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
	for _, it := range out {
		if it.ID == "" {
			t.Fatal("item ID should not be empty")
		}
	}
}

func TestPoolRecall_NilReader(t *testing.T) {
	pool := &PoolRecall{Reader: nil, SourceName: "empty"}
	rctx := &core.RecommendContext{UserID: "u1", Params: map[string]any{}}
	out, err := pool.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 0 {
		t.Fatalf("nil reader should return empty, got %d", len(out))
	}
}

func TestPoolRecall_EmptyUserID(t *testing.T) {
	reader := &fakePoolReader{items: []PoolItem{{ID: "a"}}}
	pool := &PoolRecall{Reader: reader, Strategy: NewUserShuffleStrategy()}
	rctx := &core.RecommendContext{UserID: "", Params: map[string]any{}}
	out, err := pool.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 0 {
		t.Fatalf("empty user should return empty, got %d", len(out))
	}
}

func TestPoolRecall_StrategyResolver(t *testing.T) {
	reader := &fakePoolReader{
		items: []PoolItem{
			{ID: "a", Score: 1.0},
			{ID: "b", Score: 2.0},
			{ID: "c", Score: 3.0},
		},
	}

	pool := &PoolRecall{
		Reader:     reader,
		PoolKey:    "pool:test",
		TopK:       2,
		Strategy:   NewUserShuffleStrategy(),
		SourceName: "test_pool",
		StrategyResolver: func(rctx *core.RecommendContext) DistributeStrategy {
			if rctx.Params["strategy"] == "bucket" {
				return NewUserBucketStrategy()
			}
			return nil
		},
	}

	rctx := &core.RecommendContext{
		UserID: "user-1",
		Params: map[string]any{"strategy": "bucket"},
	}
	out, err := pool.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
}

func TestPoolRecall_Name(t *testing.T) {
	pool := &PoolRecall{SourceName: "my_pool"}
	if pool.Name() != "my_pool" {
		t.Fatalf("want my_pool, got %s", pool.Name())
	}

	pool2 := &PoolRecall{}
	if pool2.Name() != "recall.pool" {
		t.Fatalf("want recall.pool, got %s", pool2.Name())
	}
}
