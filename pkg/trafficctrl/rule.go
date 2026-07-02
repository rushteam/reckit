package trafficctrl

import (
	"fmt"
	"strconv"
	"strings"
)

// Operator 规则条件运算符。
type Operator string

const (
	OpEq       Operator = "=="
	OpNe       Operator = "!="
	OpGt       Operator = ">"
	OpGte      Operator = ">="
	OpLt       Operator = "<"
	OpLte      Operator = "<="
	OpIn       Operator = "in"
	OpNotIn    Operator = "not_in"
	OpContains Operator = "contains"
)

// Rule 单条规则条件。
type Rule struct {
	Field string   `json:"field"`
	Op    Operator `json:"op"`
	Value any      `json:"value"`
}

// Match 判断给定属性值是否满足规则。
func (r *Rule) Match(attrs map[string]any) bool {
	val, ok := attrs[r.Field]
	if !ok {
		return false
	}
	return r.evaluate(val)
}

func (r *Rule) evaluate(actual any) bool {
	switch r.Op {
	case OpEq:
		return fmt.Sprint(actual) == fmt.Sprint(r.Value)
	case OpNe:
		return fmt.Sprint(actual) != fmt.Sprint(r.Value)
	case OpGt:
		return toFloat(actual) > toFloat(r.Value)
	case OpGte:
		return toFloat(actual) >= toFloat(r.Value)
	case OpLt:
		return toFloat(actual) < toFloat(r.Value)
	case OpLte:
		return toFloat(actual) <= toFloat(r.Value)
	case OpIn:
		return inSet(actual, r.Value)
	case OpNotIn:
		return !inSet(actual, r.Value)
	case OpContains:
		return strings.Contains(fmt.Sprint(actual), fmt.Sprint(r.Value))
	default:
		return false
	}
}

func toFloat(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case float32:
		return float64(n)
	case int:
		return float64(n)
	case int64:
		return float64(n)
	case int32:
		return float64(n)
	case string:
		f, _ := strconv.ParseFloat(n, 64)
		return f
	default:
		return 0
	}
}

func inSet(actual any, setVal any) bool {
	actualStr := fmt.Sprint(actual)
	switch s := setVal.(type) {
	case []any:
		for _, v := range s {
			if fmt.Sprint(v) == actualStr {
				return true
			}
		}
	case []string:
		for _, v := range s {
			if v == actualStr {
				return true
			}
		}
	}
	return false
}

// RuleSet 规则集合，所有规则取 AND 关系。
type RuleSet struct {
	Rules      []Rule `json:"rules,omitempty"`
	Expression string `json:"expression,omitempty"` // CEL 表达式（优先于 Rules）
}

// MatchAll 判断属性是否满足规则集。
// 优先级：Expression（CEL）> Rules（AND）。空规则集匹配全部。
func (rs *RuleSet) MatchAll(attrs map[string]any) bool {
	if rs.Expression != "" {
		return EvalCEL(rs.Expression, attrs)
	}
	if len(rs.Rules) == 0 {
		return true
	}
	for i := range rs.Rules {
		if !rs.Rules[i].Match(attrs) {
			return false
		}
	}
	return true
}
