# Reckit 迁移指南

本文档覆盖 reckit 最新版本中的 **Breaking Changes** 及迁移步骤。
升级后请先运行 `go build ./...`，编译器会指出所有需要修改的位置。

---

## 1. RankModel.Predict 新增 context.Context 参数

**影响**：所有实现 `model.RankModel` 接口的类型。

```go
// 旧签名
type RankModel interface {
    Name() string
    Predict(features map[string]float64) (float64, error)
}

// 新签名
type RankModel interface {
    Name() string
    Predict(ctx context.Context, features map[string]float64) (float64, error)
}
```

**迁移方法**：

```go
// 本地模型（不需要网络调用），加 _ 即可
func (m *MyLRModel) Predict(_ context.Context, features map[string]float64) (float64, error) {
    // 原有逻辑不变
}

// 远程模型，使用 ctx 传递超时/取消
func (m *MyRemoteModel) Predict(ctx context.Context, features map[string]float64) (float64, error) {
    return m.client.Score(ctx, features)
}
```

**调用方也需要更新**：

```go
// 旧
score, err := model.Predict(features)

// 新
score, err := model.Predict(ctx, features)
```

---

## 2. RPCModel.PredictBatch 新增 context.Context 参数

**影响**：直接调用 `*model.RPCModel` 的 `PredictBatch` 方法的代码。

```go
// 旧
scores, err := rpcModel.PredictBatch(featuresList)

// 新
scores, err := rpcModel.PredictBatch(ctx, featuresList)
```

通过 Pipeline Node 使用（LRNode / RPCNode / DNNNode 等）无需额外改动，ctx 已自动传递。

---

## 3. ErrorHandler.HandleError 新增 context.Context 参数

**影响**：所有实现 `recall.ErrorHandler` 接口的自定义类型。

```go
// 旧签名
type ErrorHandler interface {
    HandleError(source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error)
}

// 新签名
type ErrorHandler interface {
    HandleError(ctx context.Context, source Source, err error, rctx *core.RecommendContext) ([]*core.Item, error)
}
```

**迁移方法**：

```go
// 不需要 ctx 时
func (h *MyHandler) HandleError(_ context.Context, source recall.Source, err error, rctx *core.RecommendContext) ([]*core.Item, error) {
    // 原有逻辑不变
}

// 需要 ctx 时（如重试调用）
func (h *MyHandler) HandleError(ctx context.Context, source recall.Source, err error, rctx *core.RecommendContext) ([]*core.Item, error) {
    return source.Recall(ctx, rctx) // 使用传入的 ctx
}
```

---

## 4. 错误处理行为变更

以下组件从「静默忽略错误」改为「返回错误」，Pipeline 会中断执行。

| 组件 | 旧行为 | 新行为 |
|------|--------|--------|
| `FrequencyCapFilter` | store 出错 → 不过滤（fail-open） | 返回 error |
| `EnrichNode` | BatchGetItemFeatures 出错 → 用空特征继续 | 返回 error |
| `BlacklistFilter` | store 出错 → 不过滤 | 返回 error |

**如需保留降级行为**，在 Pipeline 中配置 `ErrorHook`：

```go
p := pipeline.New(
    pipeline.WithErrorHook(&pipeline.RecoverErrorHook{
        RecoverKinds: map[pipeline.Kind]bool{
            pipeline.KindFilter:  true,
            pipeline.KindFeature: true,
        },
    }),
)
```

这样当 Filter/Feature 阶段出错时，Pipeline 会跳过该 Node 继续执行（而非中断），同时错误会被记录。

---

## 5. EnrichNode.Kind() 返回值变更

```go
// 旧
enrich.Kind() == pipeline.KindPostProcess  // "postprocess"

// 新
enrich.Kind() == pipeline.KindFeature      // "feature"（新增常量）
```

**影响**：如果你的代码通过 `Kind()` 做路由/告警/统计，需要更新相关判断。

---

## 6. DomainError 结构体变更

新增 `Cause` 字段和 `Unwrap()` 方法：

```go
type DomainError struct {
    Code    string
    Message string
    Module  string
    Cause   error  // 新增：底层原因
}

func (e *DomainError) Unwrap() error { return e.Cause }  // 新增
```

**行为变化**：

- `Error()` 返回值：当 `Cause != nil` 时，格式变为 `"message: cause_error"`
- 支持 `errors.Is()` / `errors.As()` 链式匹配
- `IsDomainError()` / `GetDomainError()` 现在搜索整条错误链（使用 `errors.As`）

**影响**：如果你的测试通过精确字符串匹配 `err.Error()`，需要调整断言。

---

## 7. 新增公共工具函数

以下为新增 API，不影响已有代码，可选使用：

```go
// pkg/conv 包
conv.StripFeaturePrefix(features map[string]float64) map[string]float64

// pipeline 包
pipeline.KindFeature  // Kind = "feature"
```

---

## 快速迁移清单

1. `go get github.com/rushteam/reckit@latest`
2. `go build ./...` — 编译器会标出所有接口不匹配的位置
3. 搜索项目中的 `RankModel` 实现，给 `Predict` 加 `ctx context.Context` 参数
4. 搜索项目中的 `ErrorHandler` 实现，给 `HandleError` 加 `ctx context.Context` 第一个参数
5. 搜索 `PredictBatch(` 调用，加 `ctx` 参数
6. 如果之前依赖 Filter/Enrich 的 fail-open 行为，添加 `RecoverErrorHook`
7. 运行 `go test ./...` 确认所有测试通过

---

## 问题排查

| 编译错误信息 | 解决方法 |
|-------------|---------|
| `wrong type for method Predict: have Predict(map[string]float64)...` | 给你的 Predict 方法加 `ctx context.Context` 参数 |
| `wrong type for method HandleError: have HandleError(Source, error, ...)...` | 给你的 HandleError 方法加 `ctx context.Context` 第一个参数 |
| `too few arguments in call to *.PredictBatch` | 调用处第一个参数传 `ctx` |
| `undefined: pipeline.KindPostProcess` (如果你引用了) | 改为 `pipeline.KindFeature` 或保留 `KindPostProcess`（仍存在） |
