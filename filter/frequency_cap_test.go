package filter

import (
	"context"
	"testing"
	"time"

	"github.com/rushteam/reckit/core"
)

type mockCapStore struct {
	counts map[string]int
}

func (m *mockCapStore) GetImpressionCount(_ context.Context, userID, itemID string, _ time.Duration) (int, error) {
	return m.counts[userID+":"+itemID], nil
}

func TestFrequencyCapFilter(t *testing.T) {
	store := &mockCapStore{counts: map[string]int{
		"u1:a": 3,
		"u1:b": 1,
	}}
	f := &FrequencyCapFilter{Store: store, MaxCount: 3, Window: time.Hour}
	rctx := &core.RecommendContext{UserID: "u1"}

	ok1, _ := f.ShouldFilter(context.Background(), rctx, core.NewItem("a"))
	if !ok1 {
		t.Fatal("item a should be capped")
	}

	ok2, _ := f.ShouldFilter(context.Background(), rctx, core.NewItem("b"))
	if ok2 {
		t.Fatal("item b should pass")
	}
}
