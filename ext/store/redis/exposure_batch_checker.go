package redis

import (
	"bytes"
	"context"
	"fmt"
	"time"

	"github.com/bits-and-blooms/bloom/v3"
	"github.com/redis/go-redis/v9"
	"github.com/rushteam/reckit/filter"
)

const tairBloomSlotCount = 4

// LegacyBloomBatchChecker 适配 legacy STRING bloom 方案。
// 读取 rolling slot + :all 到内存后，对整批 item 做纯内存判定。
type LegacyBloomBatchChecker struct {
	Client            *redis.Client
	Capacity          uint
	FalsePositiveRate float64
}

var _ filter.BatchExposureChecker = (*LegacyBloomBatchChecker)(nil)

func NewLegacyBloomBatchChecker(client *redis.Client, capacity uint, falsePositiveRate float64) *LegacyBloomBatchChecker {
	if capacity == 0 {
		capacity = 100000
	}
	if falsePositiveRate <= 0 || falsePositiveRate >= 1 {
		falsePositiveRate = 0.01
	}
	return &LegacyBloomBatchChecker{
		Client:            client,
		Capacity:          capacity,
		FalsePositiveRate: falsePositiveRate,
	}
}

func (c *LegacyBloomBatchChecker) CheckExposedBatch(
	ctx context.Context,
	userID string,
	itemIDs []string,
	keyPrefix string,
	_ int64,
	dayWindow int,
) (map[string]bool, error) {
	out := make(map[string]bool, len(itemIDs))
	if c == nil || c.Client == nil || userID == "" || len(itemIDs) == 0 || dayWindow <= 0 {
		return out, nil
	}
	keys := rollingBloomKeys(keyPrefix, userID, dayWindow)
	blooms, err := c.preloadBlooms(ctx, keys)
	if err != nil {
		return out, err
	}
	for _, id := range itemIDs {
		b := []byte(id)
		for _, bf := range blooms {
			if bf != nil && bf.Test(b) {
				out[id] = true
				break
			}
		}
	}
	return out, nil
}

func (c *LegacyBloomBatchChecker) preloadBlooms(ctx context.Context, keys []string) ([]*bloom.BloomFilter, error) {
	pipe := c.Client.Pipeline()
	cmds := make([]*redis.StringCmd, len(keys))
	for i, key := range keys {
		cmds[i] = pipe.Get(ctx, key)
	}
	if _, err := pipe.Exec(ctx); err != nil && err != redis.Nil {
		return nil, err
	}
	blooms := make([]*bloom.BloomFilter, 0, len(keys))
	for _, cmd := range cmds {
		data, err := cmd.Bytes()
		if err != nil || len(data) == 0 {
			continue
		}
		bf := bloom.NewWithEstimates(c.Capacity, c.FalsePositiveRate)
		if _, err := bf.ReadFrom(bytes.NewReader(data)); err == nil {
			blooms = append(blooms, bf)
		}
	}
	return blooms, nil
}

// TairBloomBatchChecker 适配 TairBloom BF.MEXISTS 方案。
// 通过 pipeline 一次 round-trip 完成多 slot 检查。
type TairBloomBatchChecker struct {
	Client *redis.Client
}

var _ filter.BatchExposureChecker = (*TairBloomBatchChecker)(nil)

func NewTairBloomBatchChecker(client *redis.Client) *TairBloomBatchChecker {
	return &TairBloomBatchChecker{Client: client}
}

func (c *TairBloomBatchChecker) CheckExposedBatch(
	ctx context.Context,
	userID string,
	itemIDs []string,
	keyPrefix string,
	_ int64,
	dayWindow int,
) (map[string]bool, error) {
	out := make(map[string]bool, len(itemIDs))
	if c == nil || c.Client == nil || userID == "" || len(itemIDs) == 0 {
		return out, nil
	}
	if dayWindow <= 0 {
		dayWindow = 28
	}
	keys := rollingBloomKeys(keyPrefix, userID, dayWindow)

	pipe := c.Client.Pipeline()
	cmds := make([]*redis.Cmd, len(keys))
	for i, key := range keys {
		args := make([]interface{}, 0, len(itemIDs)+2)
		args = append(args, "BF.MEXISTS", key)
		for _, id := range itemIDs {
			args = append(args, id)
		}
		cmds[i] = pipe.Do(ctx, args...)
	}
	if _, err := pipe.Exec(ctx); err != nil && err != redis.Nil {
		return out, err
	}
	for _, cmd := range cmds {
		vals, err := cmd.Int64Slice()
		if err != nil {
			continue
		}
		for i, hit := range vals {
			if i < len(itemIDs) && hit == 1 {
				out[itemIDs[i]] = true
			}
		}
	}
	return out, nil
}

func rollingBloomKeys(keyPrefix, userID string, dayWindow int) []string {
	if keyPrefix == "" {
		keyPrefix = "user:exposed"
	}
	seen := make(map[int]struct{}, tairBloomSlotCount)
	keys := make([]string, 0, tairBloomSlotCount+1)
	now := time.Now()
	for i := 0; i < dayWindow; i++ {
		slot := weekSlot(now.AddDate(0, 0, -i))
		if _, ok := seen[slot]; ok {
			continue
		}
		seen[slot] = struct{}{}
		keys = append(keys, fmt.Sprintf("%s:bloom:%s:slot:%d", keyPrefix, userID, slot))
	}
	keys = append(keys, fmt.Sprintf("%s:bloom:%s:all", keyPrefix, userID))
	return keys
}

func weekSlot(t time.Time) int {
	weekIdx := t.Unix() / int64((7*24)*time.Hour/time.Second)
	slot := int(weekIdx % tairBloomSlotCount)
	if slot < 0 {
		slot += tairBloomSlotCount
	}
	return slot
}
