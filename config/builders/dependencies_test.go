package builders

import (
	"context"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/filter"
)

type mockStore struct{}

func (m *mockStore) Name() string                                      { return "mock" }
func (m *mockStore) Get(context.Context, string) ([]byte, error)       { return nil, core.ErrStoreNotFound }
func (m *mockStore) Set(context.Context, string, []byte, ...int) error { return nil }
func (m *mockStore) Delete(context.Context, string) error              { return nil }
func (m *mockStore) BatchGet(context.Context, []string) (map[string][]byte, error) {
	return map[string][]byte{}, nil
}
func (m *mockStore) BatchSet(context.Context, map[string][]byte, ...int) error { return nil }
func (m *mockStore) Close(context.Context) error                               { return nil }

type mockFeatureService struct{}

func (m *mockFeatureService) Name() string { return "mock_fs" }
func (m *mockFeatureService) GetUserFeatures(context.Context, string) (map[string]float64, error) {
	return map[string]float64{}, nil
}
func (m *mockFeatureService) BatchGetUserFeatures(context.Context, []string) (map[string]map[string]float64, error) {
	return map[string]map[string]float64{}, nil
}
func (m *mockFeatureService) GetItemFeatures(context.Context, string) (map[string]float64, error) {
	return map[string]float64{}, nil
}
func (m *mockFeatureService) BatchGetItemFeatures(context.Context, []string) (map[string]map[string]float64, error) {
	return map[string]map[string]float64{}, nil
}
func (m *mockFeatureService) GetRealtimeFeatures(context.Context, string, string) (map[string]float64, error) {
	return map[string]float64{}, nil
}
func (m *mockFeatureService) BatchGetRealtimeFeatures(context.Context, []core.FeatureUserItemPair) (map[core.FeatureUserItemPair]map[string]float64, error) {
	return map[core.FeatureUserItemPair]map[string]float64{}, nil
}
func (m *mockFeatureService) Close(context.Context) error { return nil }

func TestBuildFilterNode_WithInjectedStore(t *testing.T) {
	factory := NewFactory(Dependencies{
		FilterStore: &mockStore{},
	})
	cfg := map[string]interface{}{
		"filters": []interface{}{
			map[string]interface{}{"type": "blacklist", "key": "blacklist:key"},
			map[string]interface{}{"type": "user_block", "key_prefix": "user:block"},
			map[string]interface{}{"type": "exposed", "key_prefix": "user:exposed"},
		},
	}

	node, err := factory.Build("filter", cfg)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	fnode, ok := node.(*filter.FilterNode)
	if !ok {
		t.Fatalf("unexpected node type: %T", node)
	}

	bl, ok := fnode.Filters[0].(*filter.BlacklistFilter)
	if !ok || bl.Store == nil {
		t.Fatal("blacklist filter store should be injected")
	}
	ub, ok := fnode.Filters[1].(*filter.UserBlockFilter)
	if !ok || ub.Store == nil {
		t.Fatal("user block filter store should be injected")
	}
	ex, ok := fnode.Filters[2].(*filter.ExposedFilter)
	if !ok || ex.Store == nil {
		t.Fatal("exposed filter store should be injected")
	}
}

func TestBuildFeatureEnrichNode_WithInjectedFeatureService(t *testing.T) {
	factory := NewFactory(Dependencies{
		FeatureService: &mockFeatureService{},
	})
	node, err := factory.Build("feature.enrich", map[string]interface{}{})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}

	enrich, ok := node.(*feature.EnrichNode)
	if !ok {
		t.Fatalf("unexpected node type: %T", node)
	}
	if enrich.FeatureService == nil {
		t.Fatal("feature service should be injected")
	}
}
