# 配置驱动与注册规范

## 注册规范

- **NodeBuilder**：`func(config map[string]interface{}) (pipeline.Node, error)`，与 `pipeline.NodeBuilder` 一致。
- **Register(typeName, builder)**：在任意包的 `init()` 中调用 `config.Register("type.name", BuildXXX)`，该类型即可被配置驱动。
- **DefaultFactory()**：返回一个已包含所有**当前已注册** Node 类型的 `*pipeline.NodeFactory`。
- **SupportedTypes()**：返回已注册类型列表（排序），用于错误提示与校验。
- **ValidatePipelineConfig(cfg)**：校验配置中所有 node 类型均已注册；若有未支持类型，返回带「supported: [...]」的错误。

## 使用配置驱动

1. 在 main 或入口处 **import 内置 builders**（触发 init 注册）：
   ```go
   import _ "github.com/rushteam/reckit/config/builders"
   ```
2. 加载配置并构建 Pipeline：
   ```go
   cfg, _ := pipeline.LoadFromYAML("pipeline.yaml")
   if err := config.ValidatePipelineConfig(cfg); err != nil {
       // 不支持的 node 类型会在此报错，并附带 supported 列表
       return err
   }
   factory := config.DefaultFactory()
   p, err := cfg.BuildPipeline(factory)
   ```
3. 若配置中使用了未注册的 node 类型，`factory.Build(...)` 会返回错误，例如：`unknown node type "xxx" (supported: [filter feature.enrich rank.lr ...])`。

## 扩展：自定义 Node 通过 init 注册

在自定义包中实现 Builder 并在 `init()` 中注册：

```go
package mypkg

import (
    "github.com/rushteam/reckit/config"
    "github.com/rushteam/reckit/pipeline"
)

func init() {
    config.Register("my.custom.node", BuildMyNode)
}

func BuildMyNode(cfg map[string]interface{}) (pipeline.Node, error) {
    // 从 cfg 解析参数，构造并返回 pipeline.Node
    return &MyNode{}, nil
}
```

在 main 中 import 该包以触发 init：`import _ "your/mod/mypkg"`。之后 `config.DefaultFactory()` 会包含 `my.custom.node`。
