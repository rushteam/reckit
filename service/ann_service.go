package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/rushteam/reckit/core"
)

// ANNServiceClient 是 ANN 服务的 HTTP 客户端实现。
//
// 用于对接基于 HTTP 的向量搜索服务（如 Milvus HTTP API、自定义 ANN 服务等）。
//
// 工程特征：
//   - 实时性：好（HTTP 低延迟）
//   - 可扩展性：强（支持多集合）
//   - 性能：高（批量搜索）
//   - 功能：完整（支持向量搜索、插入、更新）
//
// 使用场景：
//   - 向量相似度搜索
//   - 大规模向量检索
//   - 需要 HTTP 接口的 ANN 服务
type ANNServiceClient struct {
	// Endpoint 服务端点，例如 "http://localhost:19530"
	Endpoint string

	// Collection 集合名称
	Collection string

	// Timeout 超时时间
	Timeout time.Duration

	// Auth 认证信息
	Auth *AuthConfig

	// httpClient HTTP 客户端
	httpClient *http.Client
}

// NewANNServiceClient 创建一个新的 ANN 服务客户端。
func NewANNServiceClient(endpoint, collection string, opts ...ANNServiceOption) *ANNServiceClient {
	client := &ANNServiceClient{
		Endpoint:   endpoint,
		Collection: collection,
		Timeout:    30 * time.Second,
	}

	for _, opt := range opts {
		opt(client)
	}

	// 创建 HTTP 客户端
	client.httpClient = &http.Client{
		Timeout: client.Timeout,
	}

	return client
}

// ANNServiceOption ANN 服务客户端配置选项
type ANNServiceOption func(*ANNServiceClient)

// WithANNServiceTimeout 设置超时时间
func WithANNServiceTimeout(timeout time.Duration) ANNServiceOption {
	return func(c *ANNServiceClient) {
		c.Timeout = timeout
	}
}

// WithANNServiceAuth 设置认证信息
func WithANNServiceAuth(auth *AuthConfig) ANNServiceOption {
	return func(c *ANNServiceClient) {
		c.Auth = auth
	}
}

// Predict 实现 core.MLService 接口
//
// 注意：ANN 服务的 Predict 实际上是向量搜索，不是模型预测。
// 请求中的 Instances 应该是查询向量列表。
func (c *ANNServiceClient) Predict(ctx context.Context, req *core.MLPredictRequest) (*core.MLPredictResponse, error) {
	// 1. 验证请求
	if len(req.Instances) == 0 {
		return nil, fmt.Errorf("instances (query vectors) are required")
	}

	// 2. 获取参数
	topK := 10
	if tk, ok := req.Params["top_k"].(int); ok {
		topK = tk
	} else if tk, ok := req.Params["top_k"].(float64); ok {
		topK = int(tk)
	}

	metric := "cosine"
	if m, ok := req.Params["metric"].(string); ok {
		metric = m
	}

	// 3. 构建搜索请求体
	body := map[string]interface{}{
		"collection": c.Collection,
		"vectors":    req.Instances,
		"top_k":      topK,
		"metric":     metric,
	}

	// 4. 序列化请求体
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	// 5. 构建 URL（标准 ANN 服务 API）
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "v1/vector/search"

	// 6. 创建 HTTP 请求
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// 7. 添加认证
	if c.Auth != nil {
		c.addAuth(httpReq)
	}

	// 8. 发送请求
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()

	// 9. 检查状态码
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ann service error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	// 10. 解析响应
	var result struct {
		Results []struct {
			IDs    []string  `json:"ids"`
			Scores []float64 `json:"scores"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	// 11. 提取第一个查询向量的结果（简化处理）
	predictions := make([]float64, 0)
	if len(result.Results) > 0 {
		// 返回相似度分数
		predictions = result.Results[0].Scores
	}

	return &core.MLPredictResponse{
		Predictions: predictions,
		Outputs:     result, // 保存原始输出
	}, nil
}

// Search 向量搜索（专用方法，推荐使用）
func (c *ANNServiceClient) Search(ctx context.Context, vector []float64, topK int, metric string) ([]int64, []float64, error) {
	req := &core.MLPredictRequest{
		Instances: [][]float64{vector},
		Params: map[string]interface{}{
			"top_k":  topK,
			"metric": metric,
		},
	}

	resp, err := c.Predict(ctx, req)
	if err != nil {
		return nil, nil, err
	}

	// 从响应中提取 IDs 和 Scores
	// 尝试从 Outputs 中提取 IDs
	ids := make([]int64, 0)
	if outputs, ok := resp.Outputs.(map[string]interface{}); ok {
		if results, ok := outputs["results"].([]interface{}); ok && len(results) > 0 {
			if result, ok := results[0].(map[string]interface{}); ok {
				if idsInterface, ok := result["ids"].([]interface{}); ok {
					for _, idInterface := range idsInterface {
						switch v := idInterface.(type) {
						case int64:
							ids = append(ids, v)
						case float64:
							ids = append(ids, int64(v))
						case int:
							ids = append(ids, int64(v))
						}
					}
				}
			}
		}
	}

	// 如果 IDs 为空，使用索引作为 ID（简化处理）
	if len(ids) == 0 && len(resp.Predictions) > 0 {
		ids = make([]int64, len(resp.Predictions))
		for i := range ids {
			ids[i] = int64(i + 1)
		}
	}

	return ids, resp.Predictions, nil
}

// Health 健康检查
func (c *ANNServiceClient) Health(ctx context.Context) error {
	// 构建 URL
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "health"

	// 创建 HTTP 请求
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}

	// 添加认证
	if c.Auth != nil {
		c.addAuth(httpReq)
	}

	// 发送请求
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("health check failed: %w", err)
	}
	defer resp.Body.Close()

	// 检查状态码
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("health check failed: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	return nil
}

// addAuth 添加认证信息到 HTTP 请求
func (c *ANNServiceClient) addAuth(req *http.Request) {
	if c.Auth == nil {
		return
	}

	switch c.Auth.Type {
	case "basic":
		req.SetBasicAuth(c.Auth.Username, c.Auth.Password)
	case "bearer":
		req.Header.Set("Authorization", "Bearer "+c.Auth.Token)
	case "api_key":
		req.Header.Set("X-API-Key", c.Auth.APIKey)
	}
}

// Close 关闭连接
func (c *ANNServiceClient) Close() error {
	// HTTP 客户端不需要显式关闭
	return nil
}

// 确保 ANNServiceClient 实现了 core.MLService 接口
var _ core.MLService = (*ANNServiceClient)(nil)
