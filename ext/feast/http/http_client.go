package http

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/rushteam/reckit/feast"
)

// HTTPClient 是 Feast Feature Store 的 HTTP 客户端实现。
//
// 注意：此实现位于扩展包中，需要单独引入：
//   go get github.com/rushteam/reckit/ext/feast/http
type HTTPClient struct {
	// Endpoint 服务端点，例如 "http://localhost:6566"
	Endpoint string

	// Project 项目名称
	Project string

	// Timeout 超时时间
	Timeout time.Duration

	// Auth 认证信息
	Auth *feast.AuthConfig

	// httpClient HTTP 客户端
	httpClient *http.Client
}

// NewHTTPClient 创建一个新的 Feast HTTP 客户端。
func NewHTTPClient(endpoint, project string, opts ...feast.ClientOption) (*HTTPClient, error) {
	config := &feast.ClientConfig{
		Endpoint: endpoint,
		Project:  project,
		Timeout:  30 * time.Second,
		UseGRPC:  false,
	}

	for _, opt := range opts {
		opt(config)
	}

	client := &HTTPClient{
		Endpoint:   config.Endpoint,
		Project:    config.Project,
		Timeout:    config.Timeout,
		Auth:       config.Auth,
		httpClient: &http.Client{Timeout: config.Timeout},
	}

	return client, nil
}

// GetOnlineFeatures 获取在线特征
func (c *HTTPClient) GetOnlineFeatures(ctx context.Context, req *feast.GetOnlineFeaturesRequest) (*feast.GetOnlineFeaturesResponse, error) {
	// 1. 验证请求
	if len(req.Features) == 0 {
		return nil, fmt.Errorf("features are required")
	}
	if len(req.EntityRows) == 0 {
		return nil, fmt.Errorf("entity rows are required")
	}

	// 2. 构建请求体（Feast HTTP API 格式）
	body := map[string]interface{}{
		"features":          req.Features,
		"entities":          req.EntityRows,
		"full_feature_names": false,
	}
	if req.Project != "" {
		body["project"] = req.Project
	} else if c.Project != "" {
		body["project"] = c.Project
	}

	// 3. 序列化请求体
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	// 4. 构建 URL（Feast HTTP API）
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "get-online-features"

	// 5. 创建 HTTP 请求
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// 6. 添加认证
	if c.Auth != nil {
		c.addAuth(httpReq)
	}

	// 7. 发送请求
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()

	// 8. 检查状态码
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("feast error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	// 9. 解析响应
	var result struct {
		Metadata map[string]interface{} `json:"metadata"`
		Results  []struct {
			Values map[string]interface{} `json:"values"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	// 10. 转换响应格式
	featureVectors := make([]feast.FeatureVector, len(result.Results))
	for i, r := range result.Results {
		featureVectors[i] = feast.FeatureVector{
			Values:    r.Values,
			EntityRow: req.EntityRows[i],
		}
	}

	return &feast.GetOnlineFeaturesResponse{
		FeatureVectors: featureVectors,
		Metadata:       result.Metadata,
	}, nil
}

// GetHistoricalFeatures 获取历史特征
func (c *HTTPClient) GetHistoricalFeatures(ctx context.Context, req *feast.GetHistoricalFeaturesRequest) (*feast.GetHistoricalFeaturesResponse, error) {
	// 1. 验证请求
	if len(req.Features) == 0 {
		return nil, fmt.Errorf("features are required")
	}
	if len(req.EntityDF) == 0 {
		return nil, fmt.Errorf("entity df is required")
	}

	// 2. 构建请求体（Feast HTTP API 格式）
	body := map[string]interface{}{
		"features":  req.Features,
		"entity_df": req.EntityDF,
	}
	if req.Project != "" {
		body["project"] = req.Project
	} else if c.Project != "" {
		body["project"] = c.Project
	}
	if req.StartTime != nil {
		body["start_time"] = req.StartTime.Format(time.RFC3339)
	}
	if req.EndTime != nil {
		body["end_time"] = req.EndTime.Format(time.RFC3339)
	}

	// 3. 序列化请求体
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	// 4. 构建 URL
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "get-historical-features"

	// 5. 创建 HTTP 请求
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// 6. 添加认证
	if c.Auth != nil {
		c.addAuth(httpReq)
	}

	// 7. 发送请求
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()

	// 8. 检查状态码
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("feast error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	// 9. 解析响应
	var result struct {
		DataFrame []map[string]interface{} `json:"dataframe"`
		Metadata  map[string]interface{}   `json:"metadata"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &feast.GetHistoricalFeaturesResponse{
		DataFrame: result.DataFrame,
		Metadata:  result.Metadata,
	}, nil
}

// Materialize 将特征物化到在线存储
func (c *HTTPClient) Materialize(ctx context.Context, req *feast.MaterializeRequest) error {
	// 1. 构建请求体
	body := map[string]interface{}{
		"start_time": req.StartTime.Format(time.RFC3339),
		"end_time":   req.EndTime.Format(time.RFC3339),
	}
	if req.Project != "" {
		body["project"] = req.Project
	} else if c.Project != "" {
		body["project"] = c.Project
	}
	if len(req.FeatureViews) > 0 {
		body["feature_views"] = req.FeatureViews
	}

	// 2. 序列化请求体
	jsonData, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	// 3. 构建 URL
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "materialize"

	// 4. 创建 HTTP 请求
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// 5. 添加认证
	if c.Auth != nil {
		c.addAuth(httpReq)
	}

	// 6. 发送请求
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()

	// 7. 检查状态码
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("feast error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	return nil
}

// ListFeatures 列出所有可用的特征
func (c *HTTPClient) ListFeatures(ctx context.Context) ([]feast.Feature, error) {
	// 1. 构建 URL
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "list-features"
	if c.Project != "" {
		url += "?project=" + c.Project
	}

	// 2. 创建 HTTP 请求
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// 3. 添加认证
	if c.Auth != nil {
		c.addAuth(httpReq)
	}

	// 4. 发送请求
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()

	// 5. 检查状态码
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("feast error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	// 6. 解析响应
	var result struct {
		Features []feast.Feature `json:"features"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return result.Features, nil
}

// GetFeatureService 获取特征服务信息
func (c *HTTPClient) GetFeatureService(ctx context.Context) (*feast.FeatureServiceInfo, error) {
	// 1. 构建 URL
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "info"
	if c.Project != "" {
		url += "?project=" + c.Project
	}

	// 2. 创建 HTTP 请求
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}

	// 3. 添加认证
	if c.Auth != nil {
		c.addAuth(httpReq)
	}

	// 4. 发送请求
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()

	// 5. 检查状态码
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("feast error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	// 6. 解析响应
	var info feast.FeatureServiceInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &info, nil
}

// Close 关闭连接
func (c *HTTPClient) Close() error {
	// HTTP 客户端不需要显式关闭
	return nil
}

// addAuth 添加认证头
func (c *HTTPClient) addAuth(req *http.Request) {
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

var _ feast.Client = (*HTTPClient)(nil)