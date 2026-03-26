package rank

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/model"
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

func TestLRNode_EmitExplainLabels(t *testing.T) {
	node := &LRNode{
		Model: &model.LRModel{
			Bias:    0.0,
			Weights: map[string]float64{"ctr": 1.0, "cvr": 2.0},
		},
		Explain: &LRExplainConfig{
			EmitRawScore:        true,
			EmitMissingFlag:     true,
			EmitFeatureCoverage: true,
		},
	}
	item := core.NewItem("a")
	item.Features = map[string]float64{
		"ctr":                   0.2,
		"item_features_missing": 1,
	}
	out, err := node.Process(context.Background(), &core.RecommendContext{}, []*core.Item{item})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("unexpected output length: %d", len(out))
	}
	labels := out[0].Labels
	if _, ok := labels[LabelKeyRankLinearRaw]; !ok {
		t.Fatalf("missing %s", LabelKeyRankLinearRaw)
	}
	if v := labels[LabelKeyRankFeaturesMissing].Value; v != "1" {
		t.Fatalf("want missing=1, got %s", v)
	}
	if _, ok := labels[LabelKeyRankFeatureCoverage]; !ok {
		t.Fatalf("missing %s", LabelKeyRankFeatureCoverage)
	}
}
