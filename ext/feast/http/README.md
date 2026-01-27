# Feast HTTP 扩展包

Feast HTTP 客户端实现，位于扩展包中，独立管理依赖。

## 架构说明

**领域层接口**：`feature.FeatureService`（核心包）
- 这是推荐使用的接口，位于领域层
- 所有特征服务都应实现此接口

**基础设施层接口**：`ext/feast/http.Client`（扩展包）
- 这是 Feast 特定的接口，位于基础设施层
- 主要用于适配器内部使用

**推荐使用方式**：通过适配器将 Feast 适配为 `feature.FeatureService`

## 安装

```bash
go get github.com/rushteam/reckit/ext/feast/http
```

## 使用方式

### 方式 1：通过适配器使用（推荐）

```go
import (
    "github.com/rushteam/reckit/feature"
    feasthttp "github.com/rushteam/reckit/ext/feast/http"
)

// 1. 创建 Feast HTTP 客户端（基础设施层）
feastClient, err := feasthttp.NewClient("http://localhost:6566", "my_project")
if err != nil {
    log.Fatal(err)
}
defer feastClient.Close()

// 2. 创建特征映射配置
mapping := &feasthttp.FeatureMapping{
    UserFeatures: []string{"user_stats:age", "user_stats:gender"},
    ItemFeatures: []string{"item_stats:price", "item_stats:category"},
    UserEntityKey: "user_id",
    ItemEntityKey: "item_id",
}

// 3. 创建适配器（将 Feast 适配为 feature.FeatureService）
featureService := feasthttp.NewFeatureServiceAdapter(feastClient, mapping)

// 4. 作为 feature.FeatureService 使用（领域层接口）
var fs feature.FeatureService = featureService
```

### 方式 2：直接使用 Feast 客户端（不推荐）

```go
import feasthttp "github.com/rushteam/reckit/ext/feast/http"

// 直接使用基础设施层接口（不推荐，应使用领域层接口）
client, err := feasthttp.NewClient("http://localhost:6566", "my_project")
var c feasthttp.Client = client
```

## 依赖

- `github.com/rushteam/reckit` - 核心包（领域层接口 `feature.FeatureService`）

## 自行实现

你也可以参考此实现，自行实现 `feature.FeatureService` 接口，满足你的特定需求。