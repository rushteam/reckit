# Redis Store 扩展包

Redis 存储实现，位于扩展包中，独立管理依赖。

## 安装

```bash
go get github.com/rushteam/reckit/ext/store/redis
```

## 基础使用

```go
import (
    "github.com/rushteam/reckit/core"
    redisstore "github.com/rushteam/reckit/ext/store/redis"
)

// 创建 Redis 存储
store, err := redisstore.NewRedisStore("localhost:6379", 0)
if err != nil {
    log.Fatal(err)
}
defer store.Close(ctx)

// 作为 core.Store 使用
var s core.Store = store
```

## 布隆过滤器支持

本扩展包提供了基于 Redis + [bits-and-blooms/bloom](https://github.com/bits-and-blooms/bloom) 的布隆过滤器实现，用于曝光过滤器的较长周期数据检查。

### 使用布隆过滤器

```go
import (
    "context"
    "github.com/rushteam/reckit/core"
    "github.com/rushteam/reckit/filter"
    redisstore "github.com/rushteam/reckit/ext/store/redis"
)

ctx := context.Background()

// 1. 创建 Redis 存储
store, err := redisstore.NewRedisStore("localhost:6379", 0)
if err != nil {
    log.Fatal(err)
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
    30,        // 布隆过滤器：30天（较长周期数据）
)

// 5. 使用曝光过滤器
rctx := &core.RecommendContext{
    UserID: "user_123",
    Scene:  "feed",
}

item := core.NewItem("item_456")
shouldFilter, _ := exposedFilter.ShouldFilter(ctx, rctx, item)
```

### 添加曝光数据到布隆过滤器

```go
import (
    "context"
    "fmt"
    "time"
    redisstore "github.com/rushteam/reckit/ext/store/redis"
)

ctx := context.Background()

// 创建 Redis 存储和布隆过滤器检查器
store, _ := redisstore.NewRedisStore("localhost:6379", 0)
defer store.Close(ctx)

bloomChecker := redisstore.NewRedisBloomFilterChecker(store, 1000000, 0.01)

// 获取当前日期（YYYYMMDD 格式）
now := time.Now()
dateStr := now.Format("20060102")

// key 格式：{keyPrefix}:bloom:{userID}:{date}
userID := "user_123"
itemID := "item_456"
key := fmt.Sprintf("user:exposed:bloom:%s:%s", userID, dateStr)

// 单个添加
err := bloomChecker.AddToBloomFilter(ctx, key, itemID, 0) // ttl=0 表示不过期

// 批量添加
itemIDs := []string{"item_789", "item_101", "item_202"}
err = bloomChecker.BatchAddToBloomFilter(ctx, key, itemIDs, 0)
```

### 布隆过滤器参数说明

- **capacity**: 预期容量（元素数量）
  - 例如：`1000000` 表示预期存储 100 万个元素
  - 建议根据实际业务量设置，设置过小可能导致误判率上升

- **falsePositiveRate**: 期望的误判率
  - 例如：`0.01` 表示 1% 的误判率
  - 误判率越低，所需内存越大
  - 常见值：0.01 (1%), 0.001 (0.1%)

### 布隆过滤器 Key 格式

布隆过滤器的 Redis key 格式为：`{keyPrefix}:bloom:{userID}:{date}`

- `keyPrefix`: 在创建 `ExposedFilter` 时指定的前缀（例如：`"user:exposed"`）
- `userID`: 用户 ID
- `date`: 日期，格式为 `YYYYMMDD`（例如：`20260127`）

### 性能优化

`RedisBloomFilterChecker` 内置了本地缓存机制，避免频繁从 Redis 读取和反序列化布隆过滤器数据。如果需要强制刷新缓存，可以调用：

```go
// 清除所有缓存
bloomChecker.ClearCache()

// 清除指定 key 的缓存
bloomChecker.ClearCacheKey(key)
```

## 依赖

- `github.com/rushteam/reckit` - 核心包（仅接口定义）
- `github.com/redis/go-redis/v9` - Redis 客户端
- `github.com/bits-and-blooms/bloom/v3` - 布隆过滤器实现

## 自行实现

你也可以参考此实现，自行实现 `core.Store`、`core.KeyValueStore` 或 `filter.BloomFilterChecker` 接口，满足你的特定需求。