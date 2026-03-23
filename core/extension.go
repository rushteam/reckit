package core

// Extension 是 RecommendContext 的可插拔扩展接口。
// 业务方可实现此接口，将 AB 实验、用户画像、限流状态等附加到 Context 上，
// 在 Pipeline 各 Node 中通过 ExtensionAs 类型安全地读取。
type Extension interface {
	ExtensionName() string // 全局唯一标识，如 "aippy.abtest"
}

// ExtensionAs 从 RecommendContext 中获取指定名称的 Extension，并做类型断言。
// 返回目标类型值和是否成功；未注册或类型不匹配均返回 zero value + false。
func ExtensionAs[T Extension](rctx *RecommendContext, name string) (T, bool) {
	var zero T
	e, ok := rctx.GetExtension(name)
	if !ok {
		return zero, false
	}
	v, ok := e.(T)
	return v, ok
}
