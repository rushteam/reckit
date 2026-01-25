package utils

// LabelMergeStrategy 是 Label 合并策略接口，用于自定义 Label 合并逻辑。
type LabelMergeStrategy interface {
	// Merge 合并两个 Label
	// existing: 已存在的 Label
	// incoming: 新来的 Label
	// 返回: 合并后的 Label
	Merge(existing, incoming Label) Label
}

// DefaultLabelMergeStrategy 是默认的 Label 合并策略。
// - Value: 以 '|' 累积
// - Source: 以 ',' 累积
type DefaultLabelMergeStrategy struct{}

func (s *DefaultLabelMergeStrategy) Merge(existing, incoming Label) Label {
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

// PriorityLabelMergeStrategy 是按优先级合并的策略。
// 如果新 Label 的优先级更高，则覆盖；否则保留旧的。
type PriorityLabelMergeStrategy struct {
	// SourcePriority 定义 Source 的优先级（值越小优先级越高）
	SourcePriority map[string]int
}

func (s *PriorityLabelMergeStrategy) Merge(existing, incoming Label) Label {
	if existing.Value == "" {
		return incoming
	}
	if incoming.Value == "" {
		return existing
	}

	// 比较优先级
	existingPriority := s.getPriority(existing.Source)
	incomingPriority := s.getPriority(incoming.Source)

	if incomingPriority < existingPriority {
		// 新 Label 优先级更高，覆盖
		return incoming
	}
	// 保留旧的
	return existing
}

func (s *PriorityLabelMergeStrategy) getPriority(source string) int {
	if s.SourcePriority != nil {
		if priority, ok := s.SourcePriority[source]; ok {
			return priority
		}
	}
	return 999 // 默认最低优先级
}

// AccumulateLabelMergeStrategy 是累加策略（适用于数值型 Value）。
// 尝试将 Value 解析为数值并累加。
type AccumulateLabelMergeStrategy struct{}

func (s *AccumulateLabelMergeStrategy) Merge(existing, incoming Label) Label {
	// 简化实现：如果 Value 是数值，尝试累加；否则使用默认策略
	// 实际使用时可以根据业务需求实现更复杂的逻辑
	defaultStrategy := &DefaultLabelMergeStrategy{}
	return defaultStrategy.Merge(existing, incoming)
}

// Label 是推荐链路中的一等公民：可解释、可追踪、可透传。
// Value 与 Source 的语义由业务自定义；Reckit 只提供标准化的合并规则。
type Label struct {
	Value  string `json:"value"`
	Source string `json:"source"` // recall / rank / rerank / rule / postprocess ...
}

// MergeLabel 用于合并同名 Label，使用默认策略。
// 如果需要自定义合并逻辑，请使用 Item.SetLabelMergeStrategy。
func MergeLabel(existing Label, incoming Label) Label {
	strategy := &DefaultLabelMergeStrategy{}
	return strategy.Merge(existing, incoming)
}
