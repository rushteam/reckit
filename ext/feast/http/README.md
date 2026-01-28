# Feast HTTP 子包

Feast HTTP 客户端实现，为 `ext/feast` 扩展的子包（无单独 go.mod）。

## 安装

安装整个 Feast 扩展即可使用本子包：

```bash
go get github.com/rushteam/reckit/ext/feast
```

## 使用

```go
import (
    "github.com/rushteam/reckit/core"
    feasthttp "github.com/rushteam/reckit/ext/feast/http"
    feastcommon "github.com/rushteam/reckit/ext/feast/common"
)

// 1. 创建 Feast HTTP 客户端
feastClient, err := feasthttp.NewClient("http://localhost:6566", "my_project")
if err != nil {
    log.Fatal(err)
}
defer feastClient.Close()

// 2. 创建特征映射配置（类型在 common 子包）
mapping := &feastcommon.FeatureMapping{
    UserFeatures: []string{"user_stats:age", "user_stats:gender"},
    ItemFeatures: []string{"item_stats:price", "item_stats:category"},
    UserEntityKey: "user_id",
    ItemEntityKey: "item_id",
}

// 3. 创建适配器（将 Feast 适配为 core.FeatureService）
featureService := feasthttp.NewFeatureServiceAdapter(feastClient, mapping)

// 4. 作为 core.FeatureService 使用
var fs core.FeatureService = featureService
```

## 自行实现

你也可以参考此实现，自行实现 `core.FeatureService` 接口，满足你的特定需求。
