package recall

import (
	"context"
	"testing"
)

func TestUserShuffleStrategy_Stable(t *testing.T) {
	items := []PoolItem{
		{ID: "a", Score: 1}, {ID: "b", Score: 2}, {ID: "c", Score: 3}, {ID: "d", Score: 4},
	}
	s := NewUserShuffleStrategy()
	req := DistributeRequest{UserID: "user-1", Items: items, TopK: 2}

	first, _ := s.Pick(context.Background(), req)
	second, _ := s.Pick(context.Background(), req)

	if len(first) != 2 {
		t.Fatalf("want 2, got %d", len(first))
	}
	if poolIDs(first) != poolIDs(second) {
		t.Fatalf("shuffle should be stable: %v vs %v", poolIDs(first), poolIDs(second))
	}
}

func TestUserShuffleStrategy_DifferentUsers(t *testing.T) {
	items := []PoolItem{
		{ID: "a"}, {ID: "b"}, {ID: "c"}, {ID: "d"}, {ID: "e"},
		{ID: "f"}, {ID: "g"}, {ID: "h"}, {ID: "i"}, {ID: "j"},
	}
	s := NewUserShuffleStrategy()

	r1, _ := s.Pick(context.Background(), DistributeRequest{UserID: "alice", Items: items, TopK: 5})
	r2, _ := s.Pick(context.Background(), DistributeRequest{UserID: "bob", Items: items, TopK: 5})

	if poolIDs(r1) == poolIDs(r2) {
		t.Fatal("different users should (very likely) get different results")
	}
}

func TestUserBucketStrategy_Stable(t *testing.T) {
	items := []PoolItem{
		{ID: "c", Score: 3}, {ID: "a", Score: 1}, {ID: "b", Score: 2},
	}
	s := NewUserBucketStrategy()
	req := DistributeRequest{UserID: "bucket-user", Items: items, TopK: 2}

	first, _ := s.Pick(context.Background(), req)
	second, _ := s.Pick(context.Background(), req)

	if poolIDs(first) != poolIDs(second) {
		t.Fatalf("bucket should be stable: %v vs %v", poolIDs(first), poolIDs(second))
	}
}

func TestUserBucketStrategy_Sorted(t *testing.T) {
	items := []PoolItem{{ID: "c"}, {ID: "a"}, {ID: "b"}}
	s := NewUserBucketStrategy()
	req := DistributeRequest{UserID: "u1", Items: items, TopK: 3}

	got, _ := s.Pick(context.Background(), req)
	if len(got) != 3 {
		t.Fatalf("want 3, got %d", len(got))
	}
}

type fakeExposureCounter struct {
	counts map[string]int64
	incred []string
}

func (f *fakeExposureCounter) GetCounts(_ context.Context, _ string, itemIDs []string) (map[string]int64, error) {
	out := make(map[string]int64, len(itemIDs))
	for _, id := range itemIDs {
		out[id] = f.counts[id]
	}
	return out, nil
}

func (f *fakeExposureCounter) IncrCounts(_ context.Context, _ string, itemIDs []string, _ int64) error {
	f.incred = append(f.incred, itemIDs...)
	return nil
}

func TestExposureCountStrategy_LeastExposedFirst(t *testing.T) {
	items := []PoolItem{
		{ID: "hot", Score: 99}, {ID: "cold", Score: 1}, {ID: "warm", Score: 50},
	}
	counter := &fakeExposureCounter{
		counts: map[string]int64{"hot": 100, "warm": 10, "cold": 0},
	}
	s := NewExposureCountStrategy(counter, "test:imp")
	req := DistributeRequest{UserID: "u", Items: items, TopK: 2}

	got, err := s.Pick(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if poolIDs(got) != "cold,warm" {
		t.Fatalf("want cold,warm, got %s", poolIDs(got))
	}
	if len(counter.incred) != 2 {
		t.Fatalf("RecordOnPick should record 2 items, got %d", len(counter.incred))
	}
}

func TestExposureCountStrategy_FallbackOnNilCounter(t *testing.T) {
	items := []PoolItem{{ID: "a"}, {ID: "b"}}
	s := &ExposureCountStrategy{Counter: nil, KeyPrefix: "x"}
	req := DistributeRequest{UserID: "u", Items: items, TopK: 1}

	got, err := s.Pick(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("want 1, got %d", len(got))
	}
}

func TestDistributeRequest_NormalizeTopK(t *testing.T) {
	tests := []struct {
		topK  int
		items int
		want  int
	}{
		{0, 5, 5},
		{-1, 3, 3},
		{10, 3, 3},
		{2, 5, 2},
	}
	for _, tt := range tests {
		req := DistributeRequest{
			Items: make([]PoolItem, tt.items),
			TopK:  tt.topK,
		}
		if got := req.NormalizeTopK(); got != tt.want {
			t.Errorf("NormalizeTopK(topK=%d, items=%d) = %d, want %d", tt.topK, tt.items, got, tt.want)
		}
	}
}

func poolIDs(items []PoolItem) string {
	var b []byte
	for i, it := range items {
		if i > 0 {
			b = append(b, ',')
		}
		b = append(b, it.ID...)
	}
	return string(b)
}
