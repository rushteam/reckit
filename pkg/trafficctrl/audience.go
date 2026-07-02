package trafficctrl

// Audience 人群圈选配置。
type Audience struct {
	RuleSet
}

// ItemPool 物品池配置。
type ItemPool struct {
	Type      ItemPoolType `json:"type"`
	RuleSet                // 动态规则匹配
	StaticIDs []string     `json:"static_ids,omitempty"`
}

// ItemPoolType 物品池类型。
type ItemPoolType string

const (
	ItemPoolDynamic ItemPoolType = "dynamic"
	ItemPoolStatic  ItemPoolType = "static"
	ItemPoolMixed   ItemPoolType = "mixed"
)

// ContainsID 判断给定 ID 是否在静态列表中。
func (p *ItemPool) ContainsID(id string) bool {
	for _, sid := range p.StaticIDs {
		if sid == id {
			return true
		}
	}
	return false
}

// HasStaticIDs 判断物品池是否包含静态 ID 列表。
func (p *ItemPool) HasStaticIDs() bool {
	return (p.Type == ItemPoolStatic || p.Type == ItemPoolMixed) && len(p.StaticIDs) > 0
}

// MatchItem 判断 item 属性是否满足物品池条件。
func (p *ItemPool) MatchItem(id string, attrs map[string]any) bool {
	switch p.Type {
	case ItemPoolStatic:
		return p.ContainsID(id)
	case ItemPoolDynamic:
		return p.MatchAll(attrs)
	case ItemPoolMixed:
		return p.ContainsID(id) || p.MatchAll(attrs)
	default:
		return p.MatchAll(attrs)
	}
}
