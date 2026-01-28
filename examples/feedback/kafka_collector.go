package feedback

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/twmb/franz-go/pkg/kgo"
)

// KafkaCollector Kafka 采集器（生产环境推荐）
type KafkaCollector struct {
	client        *kgo.Client
	topic         string
	batchSize     int
	flushInterval time.Duration

	mu        sync.Mutex
	buffer    []*FeedbackEvent
	lastFlush time.Time
	closed    bool
	closeOnce sync.Once
	wg        sync.WaitGroup
	stopCh    chan struct{}
}

// KafkaCollectorConfig Kafka 采集器配置
type KafkaCollectorConfig struct {
	// Kafka 配置
	Brokers []string // Kafka Broker 地址列表
	Topic   string   // Kafka Topic

	// 性能配置
	BatchSize     int           // 批量大小（建议 100-1000）
	FlushInterval time.Duration // 刷新间隔（建议 1-5 秒）

	// Kafka 客户端配置
	ClientID     string // 客户端 ID
	RequiredAcks int16  // 需要的 ACK 数量（1=leader, -1=all）
	Compression  string // 压缩类型（gzip, snappy, lz4, zstd）
	Idempotent   bool   // 是否启用幂等性
	MaxRetries   int    // 最大重试次数
}

// NewKafkaCollector 创建 Kafka 采集器
func NewKafkaCollector(config KafkaCollectorConfig) (*KafkaCollector, error) {
	if config.BatchSize <= 0 {
		config.BatchSize = 100
	}
	if config.FlushInterval <= 0 {
		config.FlushInterval = 1 * time.Second
	}
	if config.ClientID == "" {
		config.ClientID = "reckit-feedback-collector"
	}
	if config.RequiredAcks == 0 {
		config.RequiredAcks = 1 // 默认只需要 leader ACK
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}

	// 构建 franzgo 客户端选项
	opts := []kgo.Opt{
		kgo.SeedBrokers(config.Brokers...),
		kgo.ClientID(config.ClientID),
	}

	// 设置 RequiredAcks（转换为 kgo.Acks 类型）
	var acks kgo.Acks
	switch config.RequiredAcks {
	case 0:
		acks = kgo.NoAck()
	case 1:
		acks = kgo.LeaderAck()
	case -1:
		acks = kgo.AllISRAcks()
	default:
		acks = kgo.LeaderAck() // 默认使用 LeaderAck
	}
	opts = append(opts, kgo.RequiredAcks(acks))

	// 设置重试次数（使用 RecordRetries）
	if config.MaxRetries > 0 {
		opts = append(opts, kgo.RecordRetries(config.MaxRetries))
	}

	// 启用/禁用幂等性（如果配置）
	if !config.Idempotent {
		opts = append(opts, kgo.DisableIdempotentWrite())
	}
	// 默认启用幂等性，无需额外配置

	// 设置压缩
	switch config.Compression {
	case "gzip":
		opts = append(opts, kgo.ProducerBatchCompression(kgo.GzipCompression()))
	case "snappy":
		opts = append(opts, kgo.ProducerBatchCompression(kgo.SnappyCompression()))
	case "lz4":
		opts = append(opts, kgo.ProducerBatchCompression(kgo.Lz4Compression()))
	case "zstd":
		opts = append(opts, kgo.ProducerBatchCompression(kgo.ZstdCompression()))
	}

	// 创建 franzgo 客户端
	client, err := kgo.NewClient(opts...)
	if err != nil {
		return nil, err
	}

	c := &KafkaCollector{
		client:        client,
		topic:         config.Topic,
		batchSize:     config.BatchSize,
		flushInterval: config.FlushInterval,
		buffer:        make([]*FeedbackEvent, 0, config.BatchSize),
		lastFlush:     time.Now(),
		stopCh:        make(chan struct{}),
	}

	// 启动后台刷新协程
	c.wg.Add(1)
	go c.flushLoop()

	return c, nil
}

// RecordImpression 异步记录曝光（不阻塞）
func (c *KafkaCollector) RecordImpression(ctx context.Context, rctx *core.RecommendContext, items []*core.Item) error {
	if c.isClosed() {
		return nil
	}

	// 快速构建事件
	events := make([]*FeedbackEvent, 0, len(items))
	now := time.Now().Unix()

	for i, item := range items {
		event := &FeedbackEvent{
			UserID:    rctx.UserID,
			ItemID:    item.ID,
			Scene:     rctx.Scene,
			Type:      FeedbackTypeImpression,
			Timestamp: now,
			Position:  i,
			Score:     item.Score,
			Labels:    make(map[string]string),
		}

		// 提取关键标签
		if label, ok := item.Labels["recall_source"]; ok {
			event.Labels["recall_source"] = label.Value
		}
		if label, ok := item.Labels["recall_type"]; ok {
			event.Labels["recall_type"] = label.Value
		}
		if label, ok := item.Labels["recall_metric"]; ok {
			event.Labels["recall_metric"] = label.Value
		}

		events = append(events, event)
	}

	// 非阻塞写入缓冲
	return c.bufferEvents(events)
}

// RecordClick 异步记录点击
func (c *KafkaCollector) RecordClick(ctx context.Context, rctx *core.RecommendContext, itemID string, position int) error {
	if c.isClosed() {
		return nil
	}

	event := &FeedbackEvent{
		UserID:    rctx.UserID,
		ItemID:    itemID,
		Scene:     rctx.Scene,
		Type:      FeedbackTypeClick,
		Timestamp: time.Now().Unix(),
		Position:  position,
	}

	return c.bufferEvents([]*FeedbackEvent{event})
}

// RecordConversion 异步记录转化
func (c *KafkaCollector) RecordConversion(ctx context.Context, rctx *core.RecommendContext, itemID string, extras map[string]any) error {
	if c.isClosed() {
		return nil
	}

	event := &FeedbackEvent{
		UserID:    rctx.UserID,
		ItemID:    itemID,
		Scene:     rctx.Scene,
		Type:      FeedbackTypeConversion,
		Timestamp: time.Now().Unix(),
		Extras:    extras,
	}

	return c.bufferEvents([]*FeedbackEvent{event})
}

// bufferEvents 非阻塞缓冲事件
func (c *KafkaCollector) bufferEvents(events []*FeedbackEvent) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}

	c.buffer = append(c.buffer, events...)

	// 达到批量大小，触发发送
	if len(c.buffer) >= c.batchSize {
		go c.flush() // 异步发送，不阻塞
	}

	return nil
}

// flushLoop 定时刷新循环
func (c *KafkaCollector) flushLoop() {
	defer c.wg.Done()

	ticker := time.NewTicker(c.flushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.mu.Lock()
			shouldFlush := len(c.buffer) > 0 && time.Since(c.lastFlush) >= c.flushInterval
			c.mu.Unlock()

			if shouldFlush {
				c.flush()
			}
		case <-c.stopCh:
			return
		}
	}
}

// flush 刷新缓冲到 Kafka
func (c *KafkaCollector) flush() error {
	c.mu.Lock()
	if len(c.buffer) == 0 {
		c.mu.Unlock()
		return nil
	}

	// 复制缓冲并清空
	events := make([]*FeedbackEvent, len(c.buffer))
	copy(events, c.buffer)
	c.buffer = c.buffer[:0]
	c.lastFlush = time.Now()
	c.mu.Unlock()

	// 序列化并发送到 Kafka
	for _, event := range events {
		data, err := json.Marshal(event)
		if err != nil {
			// 记录错误但不阻塞（生产环境应该记录到监控系统）
			continue
		}

		// 使用 franzgo 异步发送
		record := &kgo.Record{
			Topic: c.topic,
			Key:   []byte(event.UserID), // 使用 UserID 作为 Key，保证同一用户的事件有序
			Value: data,
		}

		// 异步发送，不阻塞
		c.client.Produce(context.Background(), record, func(r *kgo.Record, err error) {
			if err != nil {
				// 发送失败，生产环境应该记录到监控系统或本地缓存
				// 可以考虑实现重试机制或本地落盘
			}
		})
	}

	return nil
}

// isClosed 检查是否已关闭
func (c *KafkaCollector) isClosed() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closed
}

// Close 优雅关闭（等待缓冲数据发送完成）
func (c *KafkaCollector) Close() error {
	var err error
	c.closeOnce.Do(func() {
		c.mu.Lock()
		c.closed = true
		c.mu.Unlock()

		// 停止刷新循环
		close(c.stopCh)

		// 最后一次刷新
		c.flush()

		// 等待后台协程结束
		c.wg.Wait()

		// 关闭 Kafka 客户端
		c.client.Close()
	})
	return err
}
