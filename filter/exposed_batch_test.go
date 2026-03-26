package filter

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

type mockBatchExposureChecker struct {
	exposed map[string]bool
}

func (m *mockBatchExposureChecker) CheckExposedBatch(
	_ context.Context,
	_ string,
	itemIDs []string,
	_ string,
	_ int64,
	_ int,
) (map[string]bool, error) {
	out := make(map[string]bool, len(itemIDs))
	for _, id := range itemIDs {
		out[id] = m.exposed[id]
	}
	return out, nil
}

func TestBatchExposedFilter_FilterBatch(t *testing.T) {
	f := &BatchExposedFilter{
		Checker: &mockBatchExposureChecker{
			exposed: map[string]bool{"2": true},
		},
	}
	items := []*core.Item{core.NewItem("1"), core.NewItem("2"), core.NewItem("3")}
	out, err := f.FilterBatch(context.Background(), &core.RecommendContext{UserID: "u1"}, items)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
	if out[0].ID != "1" || out[1].ID != "3" {
		t.Fatalf("unexpected output order: %s, %s", out[0].ID, out[1].ID)
	}
}
