package filter

import (
	"context"
	"errors"
	"testing"

	"github.com/rushteam/reckit/core"
)

func TestCompositeBlacklistFilter_MultipleProviders(t *testing.T) {
	f := NewCompositeBlacklistFilter(
		StaticBlacklistProvider([]string{"ban1", "ban2"}),
		StaticBlacklistProvider([]string{"ban3"}),
	)

	items := []*core.Item{
		core.NewItem("ok1"),
		core.NewItem("ban1"),
		core.NewItem("ok2"),
		core.NewItem("ban3"),
		core.NewItem("ok3"),
	}

	rctx := &core.RecommendContext{UserID: "u1", Params: map[string]any{}}
	out, err := f.FilterBatch(context.Background(), rctx, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 3 {
		t.Fatalf("want 3 after filtering, got %d", len(out))
	}
	for _, it := range out {
		if it.ID == "ban1" || it.ID == "ban3" {
			t.Fatalf("banned item %s should be filtered", it.ID)
		}
	}
}

func TestCompositeBlacklistFilter_EmptyProviders(t *testing.T) {
	f := NewCompositeBlacklistFilter()
	items := []*core.Item{core.NewItem("a"), core.NewItem("b")}

	rctx := &core.RecommendContext{UserID: "u1", Params: map[string]any{}}
	out, err := f.FilterBatch(context.Background(), rctx, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 2 {
		t.Fatalf("no providers should pass all items, got %d", len(out))
	}
}

func TestCompositeBlacklistFilter_ProviderError_Skipped(t *testing.T) {
	errProvider := BlacklistProviderFunc(func(_ context.Context, _ *core.RecommendContext) ([]string, error) {
		return nil, errors.New("redis timeout")
	})

	f := NewCompositeBlacklistFilter(
		errProvider,
		StaticBlacklistProvider([]string{"ban1"}),
	)

	items := []*core.Item{core.NewItem("ok"), core.NewItem("ban1")}
	rctx := &core.RecommendContext{UserID: "u1", Params: map[string]any{}}
	out, err := f.FilterBatch(context.Background(), rctx, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 {
		t.Fatalf("want 1 after filtering ban1, got %d", len(out))
	}
	if out[0].ID != "ok" {
		t.Fatalf("want ok, got %s", out[0].ID)
	}
}

func TestCompositeBlacklistFilter_ShouldFilter(t *testing.T) {
	f := NewCompositeBlacklistFilter(
		StaticBlacklistProvider([]string{"x", "y"}),
	)
	rctx := &core.RecommendContext{UserID: "u1", Params: map[string]any{}}

	banned, _ := f.ShouldFilter(context.Background(), rctx, core.NewItem("x"))
	if !banned {
		t.Fatal("x should be filtered")
	}

	ok, _ := f.ShouldFilter(context.Background(), rctx, core.NewItem("z"))
	if ok {
		t.Fatal("z should not be filtered")
	}
}

type fakeBlacklistStore struct {
	data map[string][]string
}

func (s *fakeBlacklistStore) GetBlacklist(_ context.Context, key string) ([]string, error) {
	return s.data[key], nil
}

func TestStoreBlacklistProvider(t *testing.T) {
	store := &fakeBlacklistStore{data: map[string][]string{
		"ban:global": {"a", "b"},
	}}

	provider := StoreBlacklistProvider(store, "ban:global")
	rctx := &core.RecommendContext{UserID: "u1", Params: map[string]any{}}
	ids, err := provider.GetBlacklistIDs(context.Background(), rctx)
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 {
		t.Fatalf("want 2, got %d", len(ids))
	}
}

func TestCompositeBlacklistFilter_ContextProvider(t *testing.T) {
	contextProvider := BlacklistProviderFunc(func(_ context.Context, rctx *core.RecommendContext) ([]string, error) {
		if bl, ok := rctx.Params["blacklist"]; ok {
			return bl.([]string), nil
		}
		return nil, nil
	})

	f := NewCompositeBlacklistFilter(
		StaticBlacklistProvider([]string{"global_ban"}),
		contextProvider,
	)

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{"blacklist": []string{"scene_ban"}},
	}
	items := []*core.Item{
		core.NewItem("ok"),
		core.NewItem("global_ban"),
		core.NewItem("scene_ban"),
	}

	out, err := f.FilterBatch(context.Background(), rctx, items)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0].ID != "ok" {
		t.Fatalf("want [ok], got %v", out)
	}
}
