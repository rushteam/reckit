package recall_test

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/recall"
)

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

type failSource struct {
	callCount atomic.Int32
	failUntil int32 // 前 N 次失败，之后成功
}

func (s *failSource) Name() string { return "fail_source" }
func (s *failSource) Recall(_ context.Context, _ *core.RecommendContext) ([]*core.Item, error) {
	n := s.callCount.Add(1)
	if n <= s.failUntil {
		return nil, errors.New("source error")
	}
	return []*core.Item{core.NewItem("ok")}, nil
}

type staticSource struct{ items []*core.Item }

func (s *staticSource) Name() string { return "static" }
func (s *staticSource) Recall(_ context.Context, _ *core.RecommendContext) ([]*core.Item, error) {
	return s.items, nil
}

// ---------------------------------------------------------------------------
// IgnoreErrorHandler
// ---------------------------------------------------------------------------

func TestIgnoreErrorHandler_OnErrorCallback(t *testing.T) {
	var reported bool
	h := &recall.IgnoreErrorHandler{
		OnError: func(src recall.Source, err error) { reported = true },
	}
	items, err := h.HandleError(&failSource{}, errors.New("err"), &core.RecommendContext{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if items != nil {
		t.Errorf("expected nil items, got %v", items)
	}
	if !reported {
		t.Error("OnError not called")
	}
}

// ---------------------------------------------------------------------------
// RetryErrorHandler
// ---------------------------------------------------------------------------

func TestRetryErrorHandler_SuccessOnSecondAttempt(t *testing.T) {
	src := &failSource{failUntil: 1} // 第一次失败（初始调用），第二次成功（重试）
	var retries int
	h := &recall.RetryErrorHandler{
		MaxRetries: 2,
		OnRetry:    func(_ recall.Source, attempt int, _ error) { retries = attempt },
	}
	items, err := h.HandleError(src, errors.New("initial"), &core.RecommendContext{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 1 || items[0].ID != "ok" {
		t.Errorf("expected [ok], got %v", items)
	}
	if retries < 1 {
		t.Error("OnRetry not called")
	}
}

func TestRetryErrorHandler_AllRetriesFail(t *testing.T) {
	src := &failSource{failUntil: 100}
	var gaveUp bool
	h := &recall.RetryErrorHandler{
		MaxRetries: 3,
		OnGiveUp:   func(_ recall.Source, _ error) { gaveUp = true },
	}
	items, err := h.HandleError(src, errors.New("initial"), &core.RecommendContext{})
	if err != nil {
		t.Fatalf("should not return error (degrade), got: %v", err)
	}
	if items != nil {
		t.Errorf("expected nil items after give up, got %v", items)
	}
	if !gaveUp {
		t.Error("OnGiveUp not called")
	}
}

// ---------------------------------------------------------------------------
// FallbackErrorHandler
// ---------------------------------------------------------------------------

func TestFallbackErrorHandler_UsesFallback(t *testing.T) {
	fallbackItems := []*core.Item{core.NewItem("fb1"), core.NewItem("fb2")}
	var reported bool
	h := &recall.FallbackErrorHandler{
		FallbackSource: &staticSource{items: fallbackItems},
		OnFallback:     func(_ recall.Source, _ error) { reported = true },
	}
	items, err := h.HandleError(&failSource{}, errors.New("err"), &core.RecommendContext{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 2 {
		t.Errorf("expected 2 fallback items, got %d", len(items))
	}
	if !reported {
		t.Error("OnFallback not called")
	}
}

func TestFallbackErrorHandler_NilFallback(t *testing.T) {
	h := &recall.FallbackErrorHandler{}
	items, err := h.HandleError(&failSource{}, errors.New("err"), &core.RecommendContext{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if items != nil {
		t.Errorf("expected nil, got %v", items)
	}
}
