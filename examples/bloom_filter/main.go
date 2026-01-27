package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/filter"
	redisstore "github.com/rushteam/reckit/ext/store/redis"
)

func main() {
	ctx := context.Background()

	// 1. 创建 Redis 存储
	store, err := redisstore.NewRedisStore("localhost:6379", 0)
	if err != nil {
		log.Fatalf("Failed to create Redis store: %v", err)
	}
	defer store.Close(ctx)

	// 2. 创建布隆过滤器检查器
	// 参数说明：
	//   - capacity: 1000000 表示预期存储 100 万个元素
	//   - falsePositiveRate: 0.01 表示 1% 的误判率
	bloomChecker := redisstore.NewRedisBloomFilterChecker(
		store,
		1000000, // 预期容量：100 万
		0.01,    // 误判率：1%
	)

	// 3. 创建 StoreAdapter（带布隆过滤器检查器）
	storeAdapter := filter.NewStoreAdapterWithBloomFilter(store, bloomChecker)

	// 4. 创建曝光过滤器
	// 参数说明：
	//   - storeAdapter: 存储适配器
	//   - "user:exposed": key 前缀
	//   - 7*24*3600: IDs 列表时间窗口（7天，秒）- 用于近期数据
	//   - 30: 布隆过滤器时间窗口（30天）- 用于较长周期数据
	exposedFilter := filter.NewExposedFilter(
		storeAdapter,
		"user:exposed",
		7*24*3600, // IDs 列表：7天（近期数据）
		30,         // 布隆过滤器：30天（较长周期数据）
	)

	// 5. 模拟添加曝光数据到布隆过滤器
	userID := "user_123"
	now := time.Now()
	dateStr := now.Format("20060102")
	key := fmt.Sprintf("user:exposed:bloom:%s:%s", userID, dateStr)

	// 添加一些曝光数据
	exposedItems := []string{"item_001", "item_002", "item_003", "item_004", "item_005"}
	err = bloomChecker.BatchAddToBloomFilter(ctx, key, exposedItems, 0)
	if err != nil {
		log.Fatalf("Failed to add items to bloom filter: %v", err)
	}
	fmt.Printf("Added items to bloom filter: %v\n", exposedItems)

	// 6. 测试曝光过滤器
	rctx := &core.RecommendContext{
		UserID: userID,
		Scene:  "feed",
	}

	// 测试已曝光的物品（应该被过滤）
	testItems := []struct {
		id           string
		shouldFilter bool
		description  string
	}{
		{"item_001", true, "已曝光的物品（在布隆过滤器中）"},
		{"item_999", false, "未曝光的物品（不在布隆过滤器中）"},
		{"item_002", true, "已曝光的物品（在布隆过滤器中）"},
	}

	fmt.Println("\n测试曝光过滤器：")
	for _, test := range testItems {
		item := core.NewItem(test.id)
		shouldFilter, err := exposedFilter.ShouldFilter(ctx, rctx, item)
		if err != nil {
			log.Printf("Error checking filter for item %s: %v", test.id, err)
			continue
		}

		status := "✓"
		if shouldFilter != test.shouldFilter {
			status = "✗"
		}
		fmt.Printf("%s Item: %s, ShouldFilter: %v, Expected: %v (%s)\n",
			status, test.id, shouldFilter, test.shouldFilter, test.description)
	}

	// 7. 演示如何添加新的曝光数据
	fmt.Println("\n添加新的曝光数据：")
	newItemID := "item_999"
	err = bloomChecker.AddToBloomFilter(ctx, key, newItemID, 0)
	if err != nil {
		log.Printf("Failed to add item to bloom filter: %v", err)
	} else {
		fmt.Printf("Added item %s to bloom filter\n", newItemID)

		// 再次检查，应该被过滤
		item := core.NewItem(newItemID)
		shouldFilter, _ := exposedFilter.ShouldFilter(ctx, rctx, item)
		fmt.Printf("Item %s should now be filtered: %v\n", newItemID, shouldFilter)
	}
}
