# Feast HTTP 扩展包

Feast HTTP 客户端实现，位于扩展包中，独立管理依赖。

## 安装

```bash
go get github.com/rushteam/reckit/ext/feast/http
```

## 使用

```go
import (
    "github.com/rushteam/reckit/feast"
    feasthttp "github.com/rushteam/reckit/ext/feast/http"
)

// 创建 HTTP 客户端
client, err := feasthttp.NewClient("http://localhost:6566", "my_project")
if err != nil {
    log.Fatal(err)
}
defer client.Close()

// 作为 feast.Client 使用
var c feast.Client = client
```

## 依赖

- `github.com/rushteam/reckit` - 核心包（仅接口定义）

## 自行实现

你也可以参考此实现，自行实现 `feast.Client` 接口，满足你的特定需求。