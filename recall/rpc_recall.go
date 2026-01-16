package recall

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pkg/utils"
)

// RPCRecall 是通过 RPC/HTTP 调用外部召回服务的召回源。
//
// 支持场景：
//   - 调用远程召回服务（Python/Java 等实现）
//   - 调用微服务架构中的召回服务
//   - 调用第三方召回 API
//
// 使用示例：
//
//	rpcRecall := &recall.RPCRecall{
//		Endpoint: "http://localhost:8080/recall",
//		Timeout:  2 * time.Second,
//		TopK:    20,
//	}
//
//	// 在 Fanout 中使用
//	fanout := &recall.Fanout{
//		Sources: []recall.Source{
//			rpcRecall,
//			&recall.Hot{IDs: []string{"1", "2", "3"}},
//		},
//	}
type RPCRecall struct {
	// Endpoint 召回服务端点，例如 "http://localhost:8080/recall"
	Endpoint string

	// Timeout 请求超时时间
	Timeout time.Duration

	// TopK 返回 TopK 个物品（可选，服务端也可以返回）
	TopK int

	// Client HTTP 客户端（可选，如果不设置则使用默认客户端）
	Client *http.Client

	// RequestBuilder 自定义请求构建器（可选）
	// 如果为 nil，则使用默认请求格式
	RequestBuilder func(ctx context.Context, rctx *core.RecommendContext, topK int) (map[string]interface{}, error)

	// ResponseParser 自定义响应解析器（可选）
	// 如果为 nil，则使用默认响应格式
	ResponseParser func(resp *http.Response) ([]*core.Item, error)
}

// NewRPCRecall 创建一个新的 RPC 召回源
func NewRPCRecall(endpoint string, timeout time.Duration) *RPCRecall {
	if timeout == 0 {
		timeout = 5 * time.Second
	}
	return &RPCRecall{
		Endpoint: endpoint,
		Timeout:  timeout,
		TopK:     20,
		Client: &http.Client{
			Timeout: timeout,
		},
	}
}

func (r *RPCRecall) Name() string {
	return "recall.rpc"
}

// Recall 实现 Source 接口，调用远程召回服务
func (r *RPCRecall) Recall(
	ctx context.Context,
	rctx *core.RecommendContext,
) ([]*core.Item, error) {
	if r.Endpoint == "" {
		return nil, fmt.Errorf("rpc recall endpoint is required")
	}

	// 确保有 HTTP 客户端
	if r.Client == nil {
		r.Client = &http.Client{
			Timeout: r.Timeout,
		}
	}

	// 构建请求
	var reqBody map[string]interface{}
	var err error

	if r.RequestBuilder != nil {
		// 使用自定义请求构建器
		reqBody, err = r.RequestBuilder(ctx, rctx, r.TopK)
		if err != nil {
			return nil, fmt.Errorf("build request: %w", err)
		}
	} else {
		// 使用默认请求格式
		reqBody = r.buildDefaultRequest(rctx, r.TopK)
	}

	// 序列化请求
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	// 创建 HTTP 请求
	req, err := http.NewRequestWithContext(ctx, "POST", r.Endpoint, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	// 发送请求
	resp, err := r.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("rpc recall call failed: %w", err)
	}
	defer resp.Body.Close()

	// 检查状态码
	if resp.StatusCode != http.StatusOK {
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("rpc recall error: status=%d, read body failed: %w", resp.StatusCode, err)
		}
		return nil, fmt.Errorf("rpc recall error: status=%d, body=%s", resp.StatusCode, string(body))
	}

	// 解析响应
	var items []*core.Item
	if r.ResponseParser != nil {
		// 使用自定义响应解析器
		items, err = r.ResponseParser(resp)
		if err != nil {
			return nil, fmt.Errorf("parse response: %w", err)
		}
	} else {
		// 使用默认响应格式
		items, err = r.parseDefaultResponse(resp)
		if err != nil {
			return nil, fmt.Errorf("parse response: %w", err)
		}
	}

	// 添加召回来源标签
	for _, it := range items {
		it.PutLabel("recall_source", utils.Label{Value: "rpc", Source: "recall"})
		it.PutLabel("recall_endpoint", utils.Label{Value: r.Endpoint, Source: "recall"})
	}

	return items, nil
}

// buildDefaultRequest 构建默认请求格式
//
// 默认请求格式（JSON）：
//
//	{
//		"user_id": "user_123",
//		"top_k": 20,
//		"context": {
//			"scene": "feed",
//			"user_profile": {...},
//			"realtime": {...}
//		}
//	}
func (r *RPCRecall) buildDefaultRequest(rctx *core.RecommendContext, topK int) map[string]interface{} {
	req := map[string]interface{}{
		"top_k": topK,
	}

	if rctx != nil {
		if rctx.UserID != "" {
			req["user_id"] = rctx.UserID
		}
		if rctx.Scene != "" {
			req["scene"] = rctx.Scene
		}
		if rctx.UserProfile != nil {
			req["user_profile"] = rctx.UserProfile
		}
		if rctx.Realtime != nil {
			req["realtime"] = rctx.Realtime
		}
		if rctx.Params != nil {
			req["params"] = rctx.Params
		}
	}

	return req
}

// parseDefaultResponse 解析默认响应格式
//
// 默认响应格式（JSON）：
//
//	{
//		"items": [
//			{"id": "item_1", "score": 0.95, "features": {...}, "meta": {...}},
//			{"id": "item_2", "score": 0.87, "features": {...}, "meta": {...}}
//		]
//	}
//
// 或者简化格式：
//
//	{
//		"item_ids": ["item_1", "item_2", "item_3"],
//		"scores": [0.95, 0.87, 0.82]
//	}
func (r *RPCRecall) parseDefaultResponse(resp *http.Response) ([]*core.Item, error) {
	var result struct {
		// 完整格式
		Items []struct {
			ID       string             `json:"id"`
			Score    float64            `json:"score"`
			Features map[string]float64 `json:"features,omitempty"`
			Meta     map[string]any     `json:"meta,omitempty"`
		} `json:"items"`

		// 简化格式
		ItemIDs []string  `json:"item_ids"`
		Scores  []float64 `json:"scores"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	// 优先使用完整格式
	if len(result.Items) > 0 {
		items := make([]*core.Item, 0, len(result.Items))
		for _, item := range result.Items {
			it := core.NewItem(item.ID)
			it.Score = item.Score
			if item.Features != nil {
				it.Features = item.Features
			}
			if item.Meta != nil {
				it.Meta = item.Meta
			}
			items = append(items, it)
		}
		return items, nil
	}

	// 使用简化格式
	if len(result.ItemIDs) > 0 {
		items := make([]*core.Item, 0, len(result.ItemIDs))
		for i, itemID := range result.ItemIDs {
			it := core.NewItem(itemID)
			if i < len(result.Scores) {
				it.Score = result.Scores[i]
			}
			items = append(items, it)
		}
		return items, nil
	}

	return []*core.Item{}, nil
}

// WithRequestBuilder 设置自定义请求构建器
func (r *RPCRecall) WithRequestBuilder(builder func(ctx context.Context, rctx *core.RecommendContext, topK int) (map[string]interface{}, error)) *RPCRecall {
	r.RequestBuilder = builder
	return r
}

// WithResponseParser 设置自定义响应解析器
func (r *RPCRecall) WithResponseParser(parser func(resp *http.Response) ([]*core.Item, error)) *RPCRecall {
	r.ResponseParser = parser
	return r
}

// WithTopK 设置 TopK
func (r *RPCRecall) WithTopK(topK int) *RPCRecall {
	r.TopK = topK
	return r
}

// WithClient 设置自定义 HTTP 客户端
func (r *RPCRecall) WithClient(client *http.Client) *RPCRecall {
	r.Client = client
	return r
}

// 确保 RPCRecall 实现了 Source 接口
var _ Source = (*RPCRecall)(nil)
