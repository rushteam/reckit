package pipeline_test

import (
	"bytes"
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type okNode struct{ name string }

func (n *okNode) Name() string        { return n.name }
func (n *okNode) Kind() pipeline.Kind  { return pipeline.KindRank }
func (n *okNode) Process(_ context.Context, _ *core.RecommendContext, items []*core.Item) ([]*core.Item, error) {
	for _, it := range items {
		it.Score += 1.0
	}
	return items, nil
}

type failNode struct {
	name string
	kind pipeline.Kind
}

func (n *failNode) Name() string        { return n.name }
func (n *failNode) Kind() pipeline.Kind  { return n.kind }
func (n *failNode) Process(_ context.Context, _ *core.RecommendContext, _ []*core.Item) ([]*core.Item, error) {
	return nil, errors.New("node failed")
}

func items(ids ...string) []*core.Item {
	out := make([]*core.Item, len(ids))
	for i, id := range ids {
		out[i] = core.NewItem(id)
	}
	return out
}

// ---------------------------------------------------------------------------
// Pipeline.Run + ErrorHook 集成测试
// ---------------------------------------------------------------------------

func TestPipeline_NoErrorHooks_FailFast(t *testing.T) {
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&failNode{name: "fail", kind: pipeline.KindReRank},
			&okNode{name: "ok"},
		},
	}
	_, err := p.Run(context.Background(), &core.RecommendContext{}, items("1"))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestPipeline_WarnAndSkipHook_RecoverAll(t *testing.T) {
	var buf bytes.Buffer
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&okNode{name: "rank"},
			&failNode{name: "diversity", kind: pipeline.KindReRank},
			&okNode{name: "topn"},
		},
		ErrorHooks: []pipeline.ErrorHook{
			&pipeline.WarnAndSkipHook{Writer: &buf},
		},
	}
	result, err := p.Run(context.Background(), &core.RecommendContext{}, items("a", "b"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result) != 2 {
		t.Fatalf("expected 2 items, got %d", len(result))
	}
	// okNode(rank) adds 1.0, diversity skipped, okNode(topn) adds 1.0 → score = 2.0
	if result[0].Score != 2.0 {
		t.Errorf("expected score 2.0, got %.1f", result[0].Score)
	}
	if !strings.Contains(buf.String(), "diversity") {
		t.Errorf("expected log about 'diversity', got: %s", buf.String())
	}
}

func TestPipeline_KindRecoveryHook_SelectiveRecovery(t *testing.T) {
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&failNode{name: "rerank_fail", kind: pipeline.KindReRank},
		},
		ErrorHooks: []pipeline.ErrorHook{
			&pipeline.KindRecoveryHook{
				RecoverKinds: map[pipeline.Kind]bool{
					pipeline.KindReRank: true,
				},
			},
		},
	}
	_, err := p.Run(context.Background(), &core.RecommendContext{}, items("1"))
	if err != nil {
		t.Fatalf("rerank should be recovered, got: %v", err)
	}

	// rank 类型不在恢复列表中，应该终止
	p2 := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&failNode{name: "rank_fail", kind: pipeline.KindRank},
		},
		ErrorHooks: []pipeline.ErrorHook{
			&pipeline.KindRecoveryHook{
				RecoverKinds: map[pipeline.Kind]bool{
					pipeline.KindReRank: true,
				},
			},
		},
	}
	_, err = p2.Run(context.Background(), &core.RecommendContext{}, items("1"))
	if err == nil {
		t.Fatal("rank should not be recovered")
	}
}

func TestPipeline_ErrorCallbackHook_ReportOnly(t *testing.T) {
	var reported []string
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&failNode{name: "bad_node", kind: pipeline.KindFilter},
		},
		ErrorHooks: []pipeline.ErrorHook{
			&pipeline.ErrorCallbackHook{
				Callback: func(_ context.Context, node pipeline.Node, err error) {
					reported = append(reported, node.Name())
				},
			},
		},
	}
	_, err := p.Run(context.Background(), &core.RecommendContext{}, items("1"))
	if err == nil {
		t.Fatal("ErrorCallbackHook should not recover")
	}
	if len(reported) != 1 || reported[0] != "bad_node" {
		t.Errorf("expected reported=[bad_node], got %v", reported)
	}
}

func TestPipeline_MultipleErrorHooks_AllCalled(t *testing.T) {
	var calls []string
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&failNode{name: "fail", kind: pipeline.KindReRank},
		},
		ErrorHooks: []pipeline.ErrorHook{
			// 上报 hook（不恢复）
			&pipeline.ErrorCallbackHook{
				Callback: func(_ context.Context, node pipeline.Node, _ error) {
					calls = append(calls, "report:"+node.Name())
				},
			},
			// 恢复 hook
			&pipeline.WarnAndSkipHook{Writer: &bytes.Buffer{}},
		},
	}
	_, err := p.Run(context.Background(), &core.RecommendContext{}, items("1"))
	if err != nil {
		t.Fatalf("should recover: %v", err)
	}
	if len(calls) != 1 || calls[0] != "report:fail" {
		t.Errorf("expected report:fail, got %v", calls)
	}
}

func TestCompositeErrorHook(t *testing.T) {
	var calls int
	hook := &pipeline.CompositeErrorHook{
		Hooks: []pipeline.ErrorHook{
			&pipeline.ErrorCallbackHook{
				Callback: func(_ context.Context, _ pipeline.Node, _ error) { calls++ },
			},
			&pipeline.WarnAndSkipHook{Writer: &bytes.Buffer{}},
			&pipeline.ErrorCallbackHook{
				Callback: func(_ context.Context, _ pipeline.Node, _ error) { calls++ },
			},
		},
	}
	node := &failNode{name: "n", kind: pipeline.KindFilter}
	recovered := hook.OnNodeError(context.Background(), &core.RecommendContext{}, node, errors.New("err"))
	if !recovered {
		t.Error("expected recovered=true (WarnAndSkip returns true)")
	}
	if calls != 2 {
		t.Errorf("expected 2 callback calls, got %d", calls)
	}
}
