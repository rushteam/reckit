package recall

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

type staticSource struct {
	name  string
	items []*core.Item
}

func (s *staticSource) Name() string { return s.name }
func (s *staticSource) Recall(_ context.Context, _ *core.RecommendContext) ([]*core.Item, error) {
	return s.items, nil
}

// prioritySource 模拟 Source 在 Recall() 里自行设置 recall_priority 的场景。
type prioritySource struct {
	name     string
	items    []*core.Item
	priority string
}

func (s *prioritySource) Name() string { return s.name }
func (s *prioritySource) Recall(_ context.Context, _ *core.RecommendContext) ([]*core.Item, error) {
	for _, it := range s.items {
		if it != nil && s.priority != "" {
			it.PutLabel("recall_priority", utils.Label{Value: s.priority, Source: "recall"})
		}
	}
	return s.items, nil
}

func TestFanout_RespectSourceRecallPriority(t *testing.T) {
	itemWithPriority := core.NewItem("x")
	itemWithout := core.NewItem("y")

	node := &Fanout{
		Sources: []Source{
			&staticSource{name: "s0", items: []*core.Item{itemWithout}},
			&prioritySource{name: "s1", items: []*core.Item{itemWithPriority}, priority: "0"},
		},
		Dedup: false,
	}

	out, err := node.Process(context.Background(), &core.RecommendContext{}, nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("expected 2 items, got %d", len(out))
	}

	for _, it := range out {
		lbl, ok := it.Labels["recall_priority"]
		if !ok {
			t.Fatalf("item %s missing recall_priority", it.ID)
		}
		switch it.ID {
		case "x":
			if lbl.Value != "0" {
				t.Errorf("item x: recall_priority = %q, want %q (Source 自行设置应被保留)", lbl.Value, "0")
			}
		case "y":
			if lbl.Value != "0" {
				t.Errorf("item y: recall_priority = %q, want %q (Fanout 索引)", lbl.Value, "0")
			}
		}
	}
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
