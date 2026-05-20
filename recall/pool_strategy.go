package recall

import (
	"context"
	"hash/fnv"
	"math/rand"
	"sort"
	"time"
)

// ExposureCounter 曝光计数读写接口，用于 ExposureCountStrategy。
type ExposureCounter interface {
	// GetCounts 批量获取 items 的曝光计数。key 为计数存储 key（含日期后缀），itemIDs 为待查询 ID。
	GetCounts(ctx context.Context, key string, itemIDs []string) (map[string]int64, error)
	// IncrCounts 批量累加曝光计数。
	IncrCounts(ctx context.Context, key string, itemIDs []string, delta int64) error
}

// ---------------------------------------------------------------------------
// UserShuffleStrategy: hash(user_id + 当天日期) → 确定性 shuffle → 取前 TopK
// ---------------------------------------------------------------------------

// UserShuffleStrategy 按用户确定性 shuffle 后取 TopK。
// 同用户同天稳定，不同用户看到不同子集。适合通用打散。
type UserShuffleStrategy struct{}

func NewUserShuffleStrategy() *UserShuffleStrategy { return &UserShuffleStrategy{} }

func (s *UserShuffleStrategy) Pick(_ context.Context, req DistributeRequest) ([]PoolItem, error) {
	topK := req.NormalizeTopK()
	if topK == 0 {
		return nil, nil
	}
	items := make([]PoolItem, len(req.Items))
	copy(items, req.Items)

	seed := userDateSeed(req.UserID)
	rng := rand.New(rand.NewSource(seed)) //nolint:gosec
	rng.Shuffle(len(items), func(i, j int) {
		items[i], items[j] = items[j], items[i]
	})
	return items[:topK], nil
}

// ---------------------------------------------------------------------------
// UserBucketStrategy: 按 hash(user+date)%N 分桶，从桶位起环形取 TopK
// ---------------------------------------------------------------------------

// UserBucketStrategy 将用户均分到池内不同位置，每个 item 获得近似 1/N 的用户流量。
// 适合公平分发场景。
type UserBucketStrategy struct{}

func NewUserBucketStrategy() *UserBucketStrategy { return &UserBucketStrategy{} }

func (s *UserBucketStrategy) Pick(_ context.Context, req DistributeRequest) ([]PoolItem, error) {
	topK := req.NormalizeTopK()
	if topK == 0 {
		return nil, nil
	}
	items := make([]PoolItem, len(req.Items))
	copy(items, req.Items)

	sort.SliceStable(items, func(i, j int) bool {
		return items[i].ID < items[j].ID
	})

	n := len(items)
	seed := userDateSeed(req.UserID)
	start := int(seed % int64(n))
	if start < 0 {
		start = -start
	}

	out := make([]PoolItem, 0, topK)
	for i := 0; i < topK; i++ {
		out = append(out, items[(start+i)%n])
	}
	return out, nil
}

// ---------------------------------------------------------------------------
// ExposureCountStrategy: 按曝光计数升序取 TopK，实现严格公平曝光
// ---------------------------------------------------------------------------

// ExposureCountStrategy 从计数器读取曝光次数，优先分发曝光最少的 item。
// 若 Counter 不可用或出错，自动回退 UserShuffle。
type ExposureCountStrategy struct {
	Counter ExposureCounter

	// KeyPrefix 曝光计数 key 前缀；最终 key = KeyPrefix:{yyyyMMdd}。
	// 为空时需由调用方通过 PoolKey 推导。
	KeyPrefix string

	// RecordOnPick 为 true 时在 Pick 成功后自动累加曝光计数。
	RecordOnPick bool

	// TZOffset 日期后缀的时区偏移（相对 UTC），默认 8h (UTC+8)。
	TZOffset time.Duration
}

func NewExposureCountStrategy(counter ExposureCounter, keyPrefix string) *ExposureCountStrategy {
	return &ExposureCountStrategy{
		Counter:      counter,
		KeyPrefix:    keyPrefix,
		RecordOnPick: true,
		TZOffset:     8 * time.Hour,
	}
}

func (s *ExposureCountStrategy) Pick(ctx context.Context, req DistributeRequest) ([]PoolItem, error) {
	topK := req.NormalizeTopK()
	if topK == 0 {
		return nil, nil
	}

	if s.Counter == nil {
		fallback := &UserShuffleStrategy{}
		return fallback.Pick(ctx, req)
	}

	ids := make([]string, 0, len(req.Items))
	for _, it := range req.Items {
		if it.ID != "" {
			ids = append(ids, it.ID)
		}
	}

	impKey := s.resolveImpKey()
	counts, err := s.Counter.GetCounts(ctx, impKey, ids)
	if err != nil {
		fallback := &UserShuffleStrategy{}
		return fallback.Pick(ctx, req)
	}

	items := make([]PoolItem, len(req.Items))
	copy(items, req.Items)
	sort.SliceStable(items, func(i, j int) bool {
		ci, cj := counts[items[i].ID], counts[items[j].ID]
		if ci != cj {
			return ci < cj
		}
		if items[i].Score != items[j].Score {
			return items[i].Score > items[j].Score
		}
		return items[i].ID < items[j].ID
	})

	picked := items[:topK]

	if s.RecordOnPick {
		pickedIDs := make([]string, 0, len(picked))
		for _, it := range picked {
			if it.ID != "" {
				pickedIDs = append(pickedIDs, it.ID)
			}
		}
		_ = s.Counter.IncrCounts(ctx, impKey, pickedIDs, 1)
	}

	return picked, nil
}

func (s *ExposureCountStrategy) resolveImpKey() string {
	prefix := s.KeyPrefix
	if prefix == "" {
		prefix = "recall:pool:imp"
	}
	offset := s.TZOffset
	if offset == 0 {
		offset = 8 * time.Hour
	}
	date := time.Now().UTC().Add(offset).Format("20060102")
	return prefix + ":" + date
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

func userDateSeed(userID string) int64 {
	today := time.Now().UTC().Format("2006-01-02")
	h := fnv.New64a()
	_, _ = h.Write([]byte(userID))
	_, _ = h.Write([]byte(today))
	return int64(h.Sum64())
}
