package redis

import (
	"context"
	"fmt"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/filter"
)

// ExampleRedisBloomFilter 展示如何使用 Redis + bits-and-blooms/bloom 实现布隆过滤器
func ExampleRedisBloomFilter() {
	ctx := context.Background()

	// 1. 创建 Redis 客户端
	redisClient, err := NewRedisStore("localhost:6379", 0)
	if err != nil {
		panic(err)
	}
	defer redisClient.Close(ctx)

	// 2. 创建布隆过滤器检查器
	// 参数说明：
	//   - capacity: 1000000 表示预期存储 100 万个元素
	//   - falsePositiveRate: 0.01 表示 1% 的误判率
	bloomChecker := NewRedisBloomFilterChecker(
		redisClient,
		1000000, // 预期容量：100 万
		0.01,    // 误判率：1%
	)

	// 3. 创建 StoreAdapter（带布隆过滤器检查器）
	storeAdapter := filter.NewStoreAdapterWithBloomFilter(redisClient, bloomChecker)

	// 4. 创建曝光过滤器
	// 参数说明：
	//   - storeAdapter: 存储适配器
	//   - "user:exposed": key 前缀
	//   - 7*24*3600: IDs 列表时间窗口（7天，秒）
	//   - 30: 布隆过滤器时间窗口（30天）
	exposedFilter := filter.NewExposedFilter(
		storeAdapter,
		"user:exposed",
		7*24*3600, // IDs 列表：7天（近期数据）
		30,         // 布隆过滤器：30天（较长周期数据）
	)

	// 5. 使用曝光过滤器
	rctx := &core.RecommendContext{
		UserID: "user_123",
		Scene:  "feed",
	}

	item := core.NewItem("item_456")
	shouldFilter, _ := exposedFilter.ShouldFilter(ctx, rctx, item)
	fmt.Printf("Should filter item: %v\n", shouldFilter)
}

// ExampleAddToBloomFilter 展示如何将曝光数据添加到布隆过滤器
func ExampleAddToBloomFilter() {
	ctx := context.Background()

	// 1. 创建 Redis 客户端
	redisClient, err := NewRedisStore("localhost:6379", 0)
	if err != nil {
		panic(err)
	}
	defer redisClient.Close(ctx)

	// 2. 创建布隆过滤器检查器
	bloomChecker := NewRedisBloomFilterChecker(
		redisClient,
		1000000,
		0.01,
	)

	// 3. 获取当前日期（YYYYMMDD 格式）
	now := time.Now()
	dateStr := now.Format("20060102")

	// 4. 添加曝光数据到布隆过滤器
	// key 格式：{keyPrefix}:bloom:{userID}:{date}
	userID := "user_123"
	itemID := "item_456"
	key := fmt.Sprintf("user:exposed:bloom:%s:%s", userID, dateStr)

	// 单个添加
	err = bloomChecker.AddToBloomFilter(ctx, key, itemID, 0) // ttl=0 表示不过期
	if err != nil {
		panic(err)
	}

	// 批量添加
	itemIDs := []string{"item_789", "item_101", "item_202"}
	err = bloomChecker.BatchAddToBloomFilter(ctx, key, itemIDs, 0)
	if err != nil {
		panic(err)
	}

	fmt.Printf("Added items to bloom filter: %s\n", key)
}
