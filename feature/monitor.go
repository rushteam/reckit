package feature

import (
	"context"
	"sync"
	"time"
)

// MemoryFeatureMonitor 是内存特征监控实现，用于监控特征使用情况、缺失率、错误率等。
// 生产环境可以使用 Prometheus、StatsD 等外部监控系统。
type MemoryFeatureMonitor struct {
	mu                sync.RWMutex
	featureStats      map[string]*FeatureStats
	featureValues     map[string][]float64 // 用于计算统计信息
	maxSamples        int                 // 每个特征保留的最大样本数
	updateInterval    time.Duration
	updateTicker      *time.Ticker
	stopUpdate        chan struct{}
}

// NewMemoryFeatureMonitor 创建内存特征监控
func NewMemoryFeatureMonitor(maxSamples int) *MemoryFeatureMonitor {
	monitor := &MemoryFeatureMonitor{
		featureStats:  make(map[string]*FeatureStats),
		featureValues: make(map[string][]float64),
		maxSamples:    maxSamples,
		updateInterval: 10 * time.Second,
		stopUpdate:    make(chan struct{}),
	}

	// 启动统计更新协程
	monitor.updateTicker = time.NewTicker(monitor.updateInterval)
	go monitor.updateStats()

	return monitor
}

func (m *MemoryFeatureMonitor) updateStats() {
	for {
		select {
		case <-m.updateTicker.C:
			m.calculateStats()
		case <-m.stopUpdate:
			m.updateTicker.Stop()
			return
		}
	}
}

func (m *MemoryFeatureMonitor) calculateStats() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for featureName, values := range m.featureValues {
		if len(values) == 0 {
			continue
		}

		stats := m.featureStats[featureName]
		if stats == nil {
			stats = &FeatureStats{
				FeatureName: featureName,
			}
			m.featureStats[featureName] = stats
		}

		// 计算统计信息
		stats.Mean = mean(values)
		stats.Std = std(values)
		stats.Min = min(values)
		stats.Max = max(values)
		stats.P50 = percentile(values, 0.5)
		stats.P95 = percentile(values, 0.95)
		stats.P99 = percentile(values, 0.99)
		stats.LastUpdateTime = time.Now()
	}
}

func (m *MemoryFeatureMonitor) RecordFeatureUsage(ctx context.Context, featureName string, value float64) {
	m.mu.Lock()
	defer m.mu.Unlock()

	stats := m.featureStats[featureName]
	if stats == nil {
		stats = &FeatureStats{
			FeatureName: featureName,
		}
		m.featureStats[featureName] = stats
	}
	stats.UsageCount++

	// 记录样本值（限制样本数量）
	values := m.featureValues[featureName]
	if len(values) >= m.maxSamples {
		// 移除最旧的样本
		values = values[1:]
	}
	values = append(values, value)
	m.featureValues[featureName] = values
}

func (m *MemoryFeatureMonitor) RecordFeatureMissing(ctx context.Context, featureName string, entityType string, entityID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	stats := m.featureStats[featureName]
	if stats == nil {
		stats = &FeatureStats{
			FeatureName: featureName,
		}
		m.featureStats[featureName] = stats
	}
	stats.MissingCount++
}

func (m *MemoryFeatureMonitor) RecordFeatureError(ctx context.Context, featureName string, err error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	stats := m.featureStats[featureName]
	if stats == nil {
		stats = &FeatureStats{
			FeatureName: featureName,
		}
		m.featureStats[featureName] = stats
	}
	stats.ErrorCount++
}

func (m *MemoryFeatureMonitor) GetFeatureStats(ctx context.Context, featureName string) (*FeatureStats, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	stats, ok := m.featureStats[featureName]
	if !ok {
		return nil, ErrFeatureNotFound
	}

	// 返回副本，避免并发修改
	return &FeatureStats{
		FeatureName:    stats.FeatureName,
		UsageCount:     stats.UsageCount,
		MissingCount:   stats.MissingCount,
		ErrorCount:     stats.ErrorCount,
		Mean:           stats.Mean,
		Std:            stats.Std,
		Min:            stats.Min,
		Max:            stats.Max,
		P50:            stats.P50,
		P95:            stats.P95,
		P99:            stats.P99,
		LastUpdateTime: stats.LastUpdateTime,
	}, nil
}

// Close 关闭监控，停止更新协程
func (m *MemoryFeatureMonitor) Close() {
	close(m.stopUpdate)
}

// 辅助函数：计算统计信息

func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func std(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	m := mean(values)
	sum := 0.0
	for _, v := range values {
		diff := v - m
		sum += diff * diff
	}
	return sum / float64(len(values))
}

func min(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	m := values[0]
	for _, v := range values[1:] {
		if v < m {
			m = v
		}
	}
	return m
}

func max(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	m := values[0]
	for _, v := range values[1:] {
		if v > m {
			m = v
		}
	}
	return m
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	// 排序后取分位数
	sorted := make([]float64, len(values))
	copy(sorted, values)
	
	// 简单冒泡排序（生产环境可以使用更高效的排序算法）
	for i := 0; i < len(sorted)-1; i++ {
		for j := 0; j < len(sorted)-i-1; j++ {
			if sorted[j] > sorted[j+1] {
				sorted[j], sorted[j+1] = sorted[j+1], sorted[j]
			}
		}
	}
	
	// 计算分位数索引
	idx := int(float64(len(sorted)) * p)
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
