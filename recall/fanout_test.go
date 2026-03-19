package recall

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
)

type staticSource struct {
	name  string
	items []*core.Item
}

func (s *staticSource) Name() string { return s.name }
func (s *staticSource) Recall(_ context.Context, _ *core.RecommendContext) ([]*core.Item, error) {
	return s.items, nil
}

func TestFanout_DefaultMergeStrategyDoesNotMutateField(t *testing.T) {
	node := &Fanout{
		Sources: []Source{
			&staticSource{
				name:  "s1",
				items: []*core.Item{core.NewItem("a")},
			},
		},
		Dedup: false,
	}

	out, err := node.Process(context.Background(), &core.RecommendContext{}, nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(out) != 1 || out[0].ID != "a" {
		t.Fatalf("unexpected output: %+v", out)
	}
	if node.MergeStrategy != nil {
		t.Fatal("merge strategy field should remain nil")
	}
}
