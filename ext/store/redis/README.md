# Redis Store 扩展包

Redis 存储实现，位于扩展包中，独立管理依赖。

## 安装

```bash
go get github.com/rushteam/reckit/ext/store/redis
```

## 使用

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
defer store.Close()

// 作为 core.Store 使用
var s core.Store = store
```

## 依赖

- `github.com/rushteam/reckit` - 核心包（仅接口定义）
- `github.com/redis/go-redis/v9` - Redis 客户端

## 自行实现

你也可以参考此实现，自行实现 `core.Store` 或 `core.KeyValueStore` 接口，满足你的特定需求。