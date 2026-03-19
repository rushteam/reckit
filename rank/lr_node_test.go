package rank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

type mockRankModel struct{}

func (m *mockRankModel) Name() string { return "mock" }
func (m *mockRankModel) Predict(_ map[string]float64) (float64, error) {
	return 1.0, nil
}

func TestLRNode_DefaultSortStrategyDoesNotMutateField(t *testing.T) {
	node := &LRNode{
		Model: &mockRankModel{},
	}
	items := []*core.Item{
		core.NewItem("a"),
	}

	out, err := node.Process(context.Background(), &core.RecommendContext{}, items)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(out) != 1 || out[0].Score != 1.0 {
		t.Fatalf("unexpected output: %+v", out)
	}
	if node.SortStrategy != nil {
		t.Fatal("sort strategy field should remain nil")
	}
}
