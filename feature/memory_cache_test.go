package feature

import (
	"context"
	"testing"
	"time"
)

func TestMemoryFeatureCache_GetReturnsCopy(t *testing.T) {
	cache := NewMemoryFeatureCache(16, time.Minute)
	t.Cleanup(func() {
		_ = cache.Close(context.Background())
	})

	cache.SetUserFeatures(context.Background(), "u1", map[string]float64{"a": 1}, time.Minute)

	got, ok := cache.GetUserFeatures(context.Background(), "u1")
	if !ok {
		t.Fatal("expected cache hit")
	}
	got["a"] = 999

	got2, ok := cache.GetUserFeatures(context.Background(), "u1")
	if !ok {
		t.Fatal("expected cache hit")
	}
	if got2["a"] != 1 {
		t.Fatalf("cache should not be mutated by caller, got %v", got2["a"])
	}
}

func TestMemoryFeatureCache_SetStoresCopy(t *testing.T) {
	cache := NewMemoryFeatureCache(16, time.Minute)
	t.Cleanup(func() {
		_ = cache.Close(context.Background())
	})

	features := map[string]float64{"x": 1}
	cache.SetItemFeatures(context.Background(), "i1", features, time.Minute)
	features["x"] = 2

	got, ok := cache.GetItemFeatures(context.Background(), "i1")
	if !ok {
		t.Fatal("expected cache hit")
	}
	if got["x"] != 1 {
		t.Fatalf("cache should keep original value, got %v", got["x"])
	}
}
