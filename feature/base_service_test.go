package feature

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/rushteam/reckit/core"
)

type errorProvider struct{}

func (p *errorProvider) Name() string { return "error_provider" }
func (p *errorProvider) GetUserFeatures(context.Context, string) (map[string]float64, error) {
	return nil, errors.New("provider failed")
}
func (p *errorProvider) BatchGetUserFeatures(context.Context, []string) (map[string]map[string]float64, error) {
	return nil, errors.New("provider failed")
}
func (p *errorProvider) GetItemFeatures(context.Context, string) (map[string]float64, error) {
	return nil, errors.New("provider failed")
}
func (p *errorProvider) BatchGetItemFeatures(context.Context, []string) (map[string]map[string]float64, error) {
	return nil, errors.New("provider failed")
}
func (p *errorProvider) GetRealtimeFeatures(context.Context, string, string) (map[string]float64, error) {
	return nil, errors.New("provider failed")
}
func (p *errorProvider) BatchGetRealtimeFeatures(context.Context, []core.FeatureUserItemPair) (map[core.FeatureUserItemPair]map[string]float64, error) {
	return nil, errors.New("provider failed")
}

type spyFallback struct {
	gotUserRctx *core.RecommendContext
	gotItem     *core.Item
}

func (f *spyFallback) GetUserFeatures(_ context.Context, _ string, rctx *core.RecommendContext) (map[string]float64, error) {
	f.gotUserRctx = rctx
	return map[string]float64{"u": 1}, nil
}

func (f *spyFallback) GetItemFeatures(_ context.Context, _ string, item *core.Item) (map[string]float64, error) {
	f.gotItem = item
	return map[string]float64{"i": 1}, nil
}

type ctxAwareCache struct {
	gotCtx context.Context
}

func (c *ctxAwareCache) GetUserFeatures(context.Context, string) (map[string]float64, bool) {
	return nil, false
}
func (c *ctxAwareCache) SetUserFeatures(context.Context, string, map[string]float64, time.Duration) {}
func (c *ctxAwareCache) GetItemFeatures(context.Context, string) (map[string]float64, bool) {
	return nil, false
}
func (c *ctxAwareCache) SetItemFeatures(context.Context, string, map[string]float64, time.Duration) {}
func (c *ctxAwareCache) InvalidateUserFeatures(context.Context, string)                             {}
func (c *ctxAwareCache) InvalidateItemFeatures(context.Context, string)                             {}
func (c *ctxAwareCache) Clear(ctx context.Context) {
	c.gotCtx = ctx
}

func TestBaseFeatureService_FallbackReceivesNonNilContextAndItem(t *testing.T) {
	fb := &spyFallback{}
	svc := NewBaseFeatureService(&errorProvider{}, WithFallback(fb))

	_, err := svc.GetUserFeatures(context.Background(), "u1")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if fb.gotUserRctx == nil || fb.gotUserRctx.UserID != "u1" {
		t.Fatal("fallback should receive non-nil RecommendContext with userID")
	}

	_, err = svc.GetItemFeatures(context.Background(), "i1")
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if fb.gotItem == nil || fb.gotItem.ID != "i1" {
		t.Fatal("fallback should receive non-nil item with itemID")
	}
}

func TestBaseFeatureService_CloseUsesPassedContext(t *testing.T) {
	cache := &ctxAwareCache{}
	svc := NewBaseFeatureService(&errorProvider{}, WithCache(cache, time.Minute))

	ctx := context.WithValue(context.Background(), "k", "v")
	if err := svc.Close(ctx); err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if cache.gotCtx != ctx {
		t.Fatal("cache.Clear should receive caller context")
	}
}
