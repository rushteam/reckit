package utils

// Label 是推荐链路中的一等公民：可解释、可追踪、可透传。
// Value 与 Source 的语义由业务自定义；Reckit 只提供标准化的合并规则。
type Label struct {
	Value  string `json:"value"`
	Source string `json:"source"` // recall / rank / rerank / rule / postprocess ...
}

// MergeLabel 用于合并同名 Label，遵循“保留历史、可追踪”的默认策略。
// - Value: 以 '|' 累积
// - Source: 以 ',' 累积
//
// 如果你需要更复杂的 DSL/优先级/覆盖规则，可以在上层封装自己的 merge 策略。
func MergeLabel(existing Label, incoming Label) Label {
	if existing.Value == "" {
		return incoming
	}
	if incoming.Value == "" {
		return existing
	}

	merged := existing
	merged.Value = existing.Value + "|" + incoming.Value
	switch {
	case existing.Source == "":
		merged.Source = incoming.Source
	case incoming.Source == "":
		merged.Source = existing.Source
	default:
		merged.Source = existing.Source + "," + incoming.Source
	}
	return merged
}
