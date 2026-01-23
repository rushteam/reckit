package feature

import (
	"context"
	"sync"
	"time"
)

// MemoryFeatureMonitor 是内存特征监控实现，用于监控特征使用情况、缺失率、错误率等。
// 生产环境可以使用 Prometheus、StatsD 等外部监控系统。
type MemoryFeatureMonitor struct {
	mu             sync.RWMutex
	featureStats   map[string]*FeatureStats
	featureValues  map[string][]float64 // 用于计算统计信息
	maxSamples     int                  // 每个特征保留的最大样本数
	updateInterval time.Duration
	updateTicker   *time.Ticker
	stopUpdate     chan struct{}
}

// NewMemoryFeatureMonitor 创建内存特征监控
func NewMemoryFeatureMonitor(maxSamples int) *MemoryFeatureMonitor {
	monitor := &MemoryFeatureMonitor{
		featureStats:   make(map[string]*FeatureStats),
		featureValues:  make(map[string][]float64),
		maxSamples:     maxSamples,
		updateInterval: 10 * time.Second,
		stopUpdate:     make(chan struct{}),
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

		// 使用统一的统计计算函数
		computedStats := ComputeStatistics(values)
		stats.Mean = computedStats.Mean
		stats.Std = computedStats.Std
		stats.Min = computedStats.Min
		stats.Max = computedStats.Max
		stats.P50 = computedStats.Median // P50 就是中位数
		stats.P95 = computedStats.P95
		stats.P99 = computedStats.P99
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
