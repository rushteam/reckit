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

// recallSourceSource 模拟 Source 在 Recall() 里自行设置 recall_source 的场景。
type recallSourceSource struct {
	name  string
	items []*core.Item
	src   string
}

func (s *recallSourceSource) Name() string { return s.name }
func (s *recallSourceSource) Recall(_ context.Context, _ *core.RecommendContext) ([]*core.Item, error) {
	for _, it := range s.items {
		if it != nil {
			it.PutLabel("recall_source", utils.Label{Value: s.src, Source: "recall"})
		}
	}
	return s.items, nil
}

// Bug #1 回归：Source 内部设置了 recall_source 后，Fanout 必须覆盖而非拼接。
func TestFanout_RecallSource_OverrideNotConcat(t *testing.T) {
	item := core.NewItem("x")
	node := &Fanout{
		Sources: []Source{
			&recallSourceSource{name: "recall.pool", items: []*core.Item{item}, src: "recall.l_new"},
		},
		Dedup: false,
	}

	out, err := node.Process(context.Background(), &core.RecommendContext{}, nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if len(out) != 1 {
		t.Fatalf("expected 1 item, got %d", len(out))
	}

	lbl := out[0].Labels["recall_source"]
	// 必须是 Fanout 覆盖后的值，不能包含 "|" 拼接
	if lbl.Value != "recall.pool" {
		t.Errorf("recall_source = %q, want %q (Fanout must override, not concat)", lbl.Value, "recall.pool")
	}
}

// Bug #2 回归：并发 Fanout 的结果顺序必须与 Sources 声明顺序一致。
func TestFanout_DeterministicOrder(t *testing.T) {
	// 运行多次确认顺序稳定
	for trial := 0; trial < 20; trial++ {
		node := &Fanout{
			Sources: []Source{
				&staticSource{name: "s0", items: []*core.Item{core.NewItem("a")}},
				&staticSource{name: "s1", items: []*core.Item{core.NewItem("b")}},
				&staticSource{name: "s2", items: []*core.Item{core.NewItem("c")}},
			},
			Dedup: false,
		}

		out, err := node.Process(context.Background(), &core.RecommendContext{}, nil)
		if err != nil {
			t.Fatalf("trial %d: unexpected err: %v", trial, err)
		}
		if len(out) != 3 {
			t.Fatalf("trial %d: expected 3 items, got %d", trial, len(out))
		}
		if out[0].ID != "a" || out[1].ID != "b" || out[2].ID != "c" {
			t.Fatalf("trial %d: order not stable, got %s,%s,%s", trial, out[0].ID, out[1].ID, out[2].ID)
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
