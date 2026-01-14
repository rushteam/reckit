# Store 抽象层

Store 是 Reckit 的存储抽象接口，统一支持多种后端（Redis、MySQL、ES、内存等），用于召回、特征读取、缓存等场景。

## 核心接口

### Store（基础接口）

```go
type Store interface {
    Name() string
    Get(ctx context.Context, key string) ([]byte, error)
    Set(ctx context.Context, key string, value []byte, ttl ...int) error
    Delete(ctx context.Context, key string) error
    BatchGet(ctx context.Context, keys []string) (map[string][]byte, error)
    BatchSet(ctx context.Context, kvs map[string][]byte, ttl ...int) error
    Close() error
}
```

### KeyValueStore（扩展接口）

支持更丰富的 KV 操作（有序集合、Hash）：

```go
type KeyValueStore interface {
    Store
    ZAdd(ctx context.Context, key string, score float64, member string) error
    ZRange(ctx context.Context, key string, start, stop int64) ([]string, error)
    ZScore(ctx context.Context, key string, member string) (float64, error)
    HGet(ctx context.Context, key, field string) ([]byte, error)
    HSet(ctx context.Context, key, field string, value []byte) error
    HGetAll(ctx context.Context, key string) (map[string][]byte, error)
}
```

## 实现示例

### MemoryStore（内存实现）

用于测试/开发/原型，支持 TTL，进程重启后数据丢失：

```go
memStore := store.NewMemoryStore()
defer memStore.Close()

// 使用 KeyValueStore 扩展功能
if kvStore, ok := memStore.(store.KeyValueStore); ok {
    kvStore.ZAdd(ctx, "hot:feed", 100.0, "1")
    members, _ := kvStore.ZRange(ctx, "hot:feed", 0, 9) // Top 10
}
```

### RedisStore（Redis 实现）

生产环境常用，支持持久化、集群、哨兵：

```go
redisStore, err := store.NewRedisStore("localhost:6379", 0)
if err != nil {
    log.Fatal(err)
}
defer redisStore.Close()

// RedisStore 实现了 KeyValueStore
kvStore := redisStore.(store.KeyValueStore)
kvStore.ZAdd(ctx, "hot:feed", 100.0, "1")
```

## 使用场景

### 1. 热门召回（Hot Recall）

从 Store 读取热门物品列表：

```go
hotRecall := &recall.Hot{
    Store: memStore,
    Key:   "hot:feed",
}
```

### 2. 特征存储

使用 Hash 存储物品特征：

```go
if kvStore, ok := store.(store.KeyValueStore); ok {
    kvStore.HSet(ctx, "item:123", "ctr", []byte("0.15"))
    kvStore.HSet(ctx, "item:123", "cvr", []byte("0.08"))
    
    features, _ := kvStore.HGetAll(ctx, "item:123")
}
```

### 3. 缓存模型预测结果

```go
store.Set(ctx, "pred:user:42:item:123", []byte("0.85"), 3600) // TTL 1小时
```

## 扩展实现

你可以实现自己的 Store 后端：

- **MySQLStore**：使用 MySQL 存储
- **ESStore**：使用 Elasticsearch 存储
- **MongoStore**：使用 MongoDB 存储
- **LocalFileStore**：使用本地文件存储

只需实现 `Store` 或 `KeyValueStore` 接口即可。
