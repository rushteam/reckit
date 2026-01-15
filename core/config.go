package core

import "time"

// RecallConfig 是召回相关的配置接口，用于提供默认值。
type RecallConfig interface {
	// DefaultTopKSimilarUsers 返回默认的 TopK 相似用户数
	DefaultTopKSimilarUsers() int
	
	// DefaultTopKItems 返回默认的 TopK 物品数
	DefaultTopKItems() int
	
	// DefaultMinCommonItems 返回默认的最小共同物品数
	DefaultMinCommonItems() int
	
	// DefaultMinCommonUsers 返回默认的最小共同用户数
	DefaultMinCommonUsers() int
	
	// DefaultTimeout 返回默认的超时时间
	DefaultTimeout() time.Duration
}

// DefaultRecallConfig 是默认的召回配置实现。
type DefaultRecallConfig struct{}

func (c *DefaultRecallConfig) DefaultTopKSimilarUsers() int {
	return 50
}

func (c *DefaultRecallConfig) DefaultTopKItems() int {
	return 20
}

func (c *DefaultRecallConfig) DefaultMinCommonItems() int {
	return 2
}

func (c *DefaultRecallConfig) DefaultMinCommonUsers() int {
	return 2
}

func (c *DefaultRecallConfig) DefaultTimeout() time.Duration {
	return 2 * time.Second
}
