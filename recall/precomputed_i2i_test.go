package recall

import (
	"context"
	"fmt"
	"testing"

	"github.com/rushteam/reckit/core"
)

// --- mock stores ---

type mockSimilarStore struct {
	data map[string][]core.ScoredMember
}

func (m *mockSimilarStore) BatchGetSimilar(_ context.Context, keys []string, topK int) ([][]core.ScoredMember, error) {
	out := make([][]core.ScoredMember, len(keys))
	for i, k := range keys {
		items := m.data[k]
		if len(items) > topK {
			items = items[:topK]
		}
		out[i] = items
	}
	return out, nil
}

type mockVersionedSimilarStore struct {
	mockSimilarStore
	version string
}

func (m *mockVersionedSimilarStore) GetActiveVersion(_ context.Context, _ string) (string, error) {
	if m.version == "" {
		return "", fmt.Errorf("version not available")
	}
	return m.version, nil
}

type mockHistoryStore struct {
	history []ScoredHistoryItem
}

func (m *mockHistoryStore) GetUserHistory(_ context.Context, _, _, _ string, _ int64) ([]ScoredHistoryItem, error) {
	return m.history, nil
}

// --- tests ---

func TestPrecomputedI2IRecall_Basic(t *testing.T) {
	store := &mockSimilarStore{
		data: map[string][]core.ScoredMember{
			"cf:similar:item_1": {
				{Member: "item_A", Score: 0.9},
				{Member: "item_B", Score: 0.8},
			},
			"cf:similar:item_2": {
				{Member: "item_A", Score: 0.7},
				{Member: "item_C", Score: 0.6},
			},
		},
	}

	r := &PrecomputedI2IRecall{
		SimilarStore: store,
		TopK:         10,
		TopKPerItem:  5,
	}

	rctx := &core.RecommendContext{
		UserID: "user_1",
		Params: map[string]any{
			"user_history": []string{"item_1", "item_2"},
		},
	}

	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) == 0 {
		t.Fatal("expected items, got none")
	}

	// item_A 应该累加分数：1.0*0.9 + 0.5*0.7 = 1.25
	// item_B: 1.0*0.8 = 0.8
	// item_C: 0.5*0.6 = 0.3
	scoreMap := make(map[string]float64)
	for _, it := range items {
		scoreMap[it.ID] = it.Score
	}

	if len(scoreMap) != 3 {
		t.Fatalf("expected 3 items, got %d", len(scoreMap))
	}
	if items[0].ID != "item_A" {
		t.Errorf("expected item_A as top item, got %s", items[0].ID)
	}
	if items[0].Score < 1.2 || items[0].Score > 1.3 {
		t.Errorf("expected item_A score ~1.25, got %f", items[0].Score)
	}
}

func TestPrecomputedI2IRecall_ExcludesHistory(t *testing.T) {
	store := &mockSimilarStore{
		data: map[string][]core.ScoredMember{
			"cf:similar:item_1": {
				{Member: "item_1", Score: 1.0}, // should be excluded
				{Member: "item_A", Score: 0.9},
			},
		},
	}

	r := &PrecomputedI2IRecall{
		SimilarStore: store,
		TopK:         10,
	}

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{
			"user_history": []string{"item_1"},
		},
	}

	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, it := range items {
		if it.ID == "item_1" {
			t.Error("history item should be excluded")
		}
	}
	if len(items) != 1 || items[0].ID != "item_A" {
		t.Errorf("expected [item_A], got %v", items)
	}
}

func TestPrecomputedI2IRecall_SkipsNegativeScores(t *testing.T) {
	store := &mockSimilarStore{
		data: map[string][]core.ScoredMember{
			"cf:similar:item_1": {
				{Member: "item_A", Score: -0.5},
				{Member: "item_B", Score: 0.0},
				{Member: "item_C", Score: 0.3},
			},
		},
	}

	r := &PrecomputedI2IRecall{SimilarStore: store, TopK: 10}

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{"user_history": []string{"item_1"}},
	}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 1 || items[0].ID != "item_C" {
		t.Errorf("expected [item_C], got %v", items)
	}
}

func TestPrecomputedI2IRecall_VersionedPrefix(t *testing.T) {
	store := &mockVersionedSimilarStore{
		mockSimilarStore: mockSimilarStore{
			data: map[string][]core.ScoredMember{
				"cf:similar:v2026-03-07:item_1": {
					{Member: "item_X", Score: 0.95},
				},
			},
		},
		version: "2026-03-07",
	}

	r := &PrecomputedI2IRecall{
		SimilarStore: store,
		VersionKey:   "cf:similar:active_version",
		TopK:         10,
	}

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{"user_history": []string{"item_1"}},
	}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 1 || items[0].ID != "item_X" {
		t.Errorf("expected [item_X], got %v", items)
	}
}

func TestPrecomputedI2IRecall_VersionUnavailable(t *testing.T) {
	store := &mockVersionedSimilarStore{version: ""}

	r := &PrecomputedI2IRecall{
		SimilarStore: store,
		VersionKey:   "some_version_key",
	}

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{"user_history": []string{"item_1"}},
	}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 0 {
		t.Error("expected empty result when version unavailable")
	}
}

func TestPrecomputedI2IRecall_QuadraticDecay(t *testing.T) {
	store := &mockSimilarStore{
		data: map[string][]core.ScoredMember{
			"cf:similar:item_1": {{Member: "item_A", Score: 1.0}},
			"cf:similar:item_2": {{Member: "item_A", Score: 1.0}},
			"cf:similar:item_3": {{Member: "item_A", Score: 1.0}},
		},
	}

	r := &PrecomputedI2IRecall{
		SimilarStore: store,
		WeightFunc:   QuadraticDecay,
		TopK:         10,
	}

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{"user_history": []string{"item_1", "item_2", "item_3"}},
	}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d", len(items))
	}
	// quadratic: 1/1 + 1/4 + 1/9 ≈ 1.3611
	expected := 1.0 + 0.25 + 1.0/9.0
	if diff := items[0].Score - expected; diff > 0.01 || diff < -0.01 {
		t.Errorf("expected score ~%.4f, got %.4f", expected, items[0].Score)
	}
}

func TestPrecomputedI2IRecall_HistoryFromStore(t *testing.T) {
	histStore := &mockHistoryStore{
		history: []ScoredHistoryItem{
			{ItemID: "item_1", Score: 1000},
			{ItemID: "item_2", Score: 900},
		},
	}
	simStore := &mockSimilarStore{
		data: map[string][]core.ScoredMember{
			"cf:similar:item_1": {{Member: "item_X", Score: 0.8}},
			"cf:similar:item_2": {{Member: "item_Y", Score: 0.7}},
		},
	}

	r := &PrecomputedI2IRecall{
		SimilarStore: simStore,
		HistoryStore: histStore,
		TopK:         10,
	}

	rctx := &core.RecommendContext{UserID: "u1"}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 items, got %d", len(items))
	}
}

func TestPrecomputedI2IRecall_HistoryFromAttributes(t *testing.T) {
	store := &mockSimilarStore{
		data: map[string][]core.ScoredMember{
			"cf:similar:item_1": {{Member: "item_Z", Score: 0.5}},
		},
	}

	r := &PrecomputedI2IRecall{
		SimilarStore: store,
		HistoryKey:   "recent_clicks",
		TopK:         10,
	}

	rctx := &core.RecommendContext{
		UserID:     "u1",
		Attributes: map[string]any{"recent_clicks": []string{"item_1"}},
	}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 1 || items[0].ID != "item_Z" {
		t.Errorf("expected [item_Z], got %v", items)
	}
}

func TestPrecomputedI2IRecall_MaxHistoryTruncation(t *testing.T) {
	data := make(map[string][]core.ScoredMember)
	history := make([]string, 100)
	for i := 0; i < 100; i++ {
		id := fmt.Sprintf("item_%d", i)
		history[i] = id
		data[fmt.Sprintf("cf:similar:%s", id)] = []core.ScoredMember{
			{Member: fmt.Sprintf("sim_%d", i), Score: 1.0},
		}
	}
	store := &mockSimilarStore{data: data}

	r := &PrecomputedI2IRecall{
		SimilarStore:    store,
		MaxHistoryItems: 5,
		TopK:            100,
	}

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{"user_history": history},
	}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// only 5 history items used → at most 5 unique sim items
	if len(items) > 5 {
		t.Errorf("expected at most 5 items (max_history=5), got %d", len(items))
	}
}

func TestPrecomputedI2IRecall_TopKTruncation(t *testing.T) {
	data := make(map[string][]core.ScoredMember)
	history := make([]string, 10)
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("item_%d", i)
		history[i] = id
		sims := make([]core.ScoredMember, 20)
		for j := 0; j < 20; j++ {
			sims[j] = core.ScoredMember{
				Member: fmt.Sprintf("sim_%d_%d", i, j),
				Score:  float64(20-j) / 20.0,
			}
		}
		data[fmt.Sprintf("cf:similar:%s", id)] = sims
	}
	store := &mockSimilarStore{data: data}

	r := &PrecomputedI2IRecall{
		SimilarStore: store,
		TopK:         3,
		TopKPerItem:  20,
	}

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{"user_history": history},
	}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 3 {
		t.Errorf("expected 3 items (top_k=3), got %d", len(items))
	}
}

func TestPrecomputedI2IRecall_EmptyHistory(t *testing.T) {
	r := &PrecomputedI2IRecall{
		SimilarStore: &mockSimilarStore{},
	}
	rctx := &core.RecommendContext{UserID: "u1"}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 0 {
		t.Error("expected empty result for empty history")
	}
}

func TestPrecomputedI2IRecall_NilStore(t *testing.T) {
	r := &PrecomputedI2IRecall{}
	rctx := &core.RecommendContext{UserID: "u1"}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 0 {
		t.Error("expected empty result for nil store")
	}
}

func TestPrecomputedI2IRecall_CustomPrefix(t *testing.T) {
	store := &mockSimilarStore{
		data: map[string][]core.ScoredMember{
			"swing:similar:item_1": {{Member: "item_X", Score: 0.9}},
		},
	}

	r := &PrecomputedI2IRecall{
		SimilarStore:     store,
		SimilarKeyPrefix: "swing:similar",
		NodeName:         "recall.swing",
		TopK:             10,
	}

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{"user_history": []string{"item_1"}},
	}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 1 || items[0].ID != "item_X" {
		t.Errorf("expected [item_X], got %v", items)
	}
	if r.Name() != "recall.swing" {
		t.Errorf("expected name recall.swing, got %s", r.Name())
	}
}

func TestPrecomputedI2IRecall_ScoreAccumulation(t *testing.T) {
	store := &mockSimilarStore{
		data: map[string][]core.ScoredMember{
			"cf:similar:item_1": {
				{Member: "item_A", Score: 0.5},
				{Member: "item_B", Score: 0.3},
			},
			"cf:similar:item_2": {
				{Member: "item_A", Score: 0.4},
				{Member: "item_C", Score: 0.6},
			},
			"cf:similar:item_3": {
				{Member: "item_A", Score: 0.3},
				{Member: "item_B", Score: 0.2},
			},
		},
	}

	r := &PrecomputedI2IRecall{
		SimilarStore: store,
		TopK:         10,
	}

	rctx := &core.RecommendContext{
		UserID: "u1",
		Params: map[string]any{"user_history": []string{"item_1", "item_2", "item_3"}},
	}
	items, err := r.Recall(context.Background(), rctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	scores := make(map[string]float64)
	for _, it := range items {
		scores[it.ID] = it.Score
	}

	// item_A: 1.0*0.5 + 0.5*0.4 + 0.333*0.3 ≈ 0.8
	expectedA := 1.0*0.5 + 0.5*0.4 + (1.0/3.0)*0.3
	if diff := scores["item_A"] - expectedA; diff > 0.01 || diff < -0.01 {
		t.Errorf("item_A: expected ~%.4f, got %.4f", expectedA, scores["item_A"])
	}

	if items[0].ID != "item_A" {
		t.Errorf("item_A should be top scored, got %s", items[0].ID)
	}
}

func TestLinearDecay(t *testing.T) {
	cases := []struct {
		index    int
		expected float64
	}{
		{0, 1.0},
		{1, 0.5},
		{2, 1.0 / 3.0},
		{9, 0.1},
	}
	for _, tc := range cases {
		got := LinearDecay(tc.index)
		if diff := got - tc.expected; diff > 1e-9 || diff < -1e-9 {
			t.Errorf("LinearDecay(%d) = %f, want %f", tc.index, got, tc.expected)
		}
	}
}

func TestQuadraticDecay(t *testing.T) {
	cases := []struct {
		index    int
		expected float64
	}{
		{0, 1.0},
		{1, 0.25},
		{2, 1.0 / 9.0},
	}
	for _, tc := range cases {
		got := QuadraticDecay(tc.index)
		if diff := got - tc.expected; diff > 1e-9 || diff < -1e-9 {
			t.Errorf("QuadraticDecay(%d) = %f, want %f", tc.index, got, tc.expected)
		}
	}
}
