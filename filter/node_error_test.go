package filter

import (
	"context"
	"errors"
	"testing"

	"github.com/rushteam/reckit/core"
)

type errBatchFilter struct {
	name string
	err  error
}

func (f *errBatchFilter) Name() string { return f.name }
func (f *errBatchFilter) ShouldFilter(_ context.Context, _ *core.RecommendContext, _ *core.Item) (bool, error) {
	return false, nil
}
func (f *errBatchFilter) FilterBatch(_ context.Context, _ *core.RecommendContext, _ []*core.Item) ([]*core.Item, error) {
	return nil, f.err
}

type errItemFilter struct {
	name string
	err  error
}

func (f *errItemFilter) Name() string { return f.name }
func (f *errItemFilter) ShouldFilter(_ context.Context, _ *core.RecommendContext, _ *core.Item) (bool, error) {
	return false, f.err
}

func TestFilterNode_BatchFilterErrorShouldReturn(t *testing.T) {
	node := &FilterNode{
		Filters: []Filter{
			&errBatchFilter{name: "batch_err", err: errors.New("batch failed")},
		},
	}

	_, err := node.Process(context.Background(), &core.RecommendContext{}, makeItems("a", "b"))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}

func TestFilterNode_ItemFilterErrorShouldReturn(t *testing.T) {
	node := &FilterNode{
		Filters: []Filter{
			&errItemFilter{name: "item_err", err: errors.New("item failed")},
		},
	}

	_, err := node.Process(context.Background(), &core.RecommendContext{}, makeItems("a"))
	if err == nil {
		t.Fatal("expected error, got nil")
	}
}
