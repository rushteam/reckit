package core

import "time"

// RecallConfig 是通用召回配置接口（所有召回算法均可使用）。
type RecallConfig interface {
	DefaultTopKItems() int
	DefaultTimeout() time.Duration
}

// CFConfig 是协同过滤专用配置接口，继承 RecallConfig 并追加 CF 特有的参数。
// 仅 UserBasedCF / ItemBasedCF 使用。
type CFConfig interface {
	RecallConfig
	DefaultTopKSimilarUsers() int
	DefaultMinCommonItems() int
	DefaultMinCommonUsers() int
}

// DefaultRecallConfig 同时实现 RecallConfig 和 CFConfig，提供合理默认值。
type DefaultRecallConfig struct{}

func (c *DefaultRecallConfig) DefaultTopKSimilarUsers() int { return 50 }
func (c *DefaultRecallConfig) DefaultTopKItems() int        { return 20 }
func (c *DefaultRecallConfig) DefaultMinCommonItems() int   { return 2 }
func (c *DefaultRecallConfig) DefaultMinCommonUsers() int   { return 2 }
func (c *DefaultRecallConfig) DefaultTimeout() time.Duration { return 2 * time.Second }
