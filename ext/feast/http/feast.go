package http

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/ext/feast/common"
)

// FeatureServiceAdapter 将 Feast Client 适配为 core.FeatureService 接口。
//
// 注意：此实现位于扩展包中，需要单独引入：
//
//	go get github.com/rushteam/reckit/ext/feast/http
//
// 这是推荐的使用方式：通过适配器将 Feast（基础设施层）适配为 core.FeatureService（领域层）。
type FeatureServiceAdapter struct {
	client         common.Client
	featureMapping *common.FeatureMapping
}

// NewFeatureServiceAdapter 创建一个新的 FeatureService 适配器。
//
// 参数：
//   - client: Feast 客户端（基础设施层）
//   - mapping: 特征映射配置
//
// 返回：
//   - *FeatureServiceAdapter: 实现了 core.FeatureService 接口的适配器
func NewFeatureServiceAdapter(client common.Client, mapping *common.FeatureMapping) *FeatureServiceAdapter {
	if mapping.UserEntityKey == "" {
		mapping.UserEntityKey = "user_id"
	}
	if mapping.ItemEntityKey == "" {
		mapping.ItemEntityKey = "item_id"
	}

	return &FeatureServiceAdapter{
		client:         client,
		featureMapping: mapping,
	}
}

// Name 返回特征服务名称
func (a *FeatureServiceAdapter) Name() string {
	return "feast"
}

// GetUserFeatures 获取用户特征
func (a *FeatureServiceAdapter) GetUserFeatures(ctx context.Context, userID string) (map[string]float64, error) {
	if len(a.featureMapping.UserFeatures) == 0 {
		return make(map[string]float64), nil
	}

	// 构建实体行
	entityRows := []map[string]interface{}{
		{a.featureMapping.UserEntityKey: userID},
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.UserFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast get user features failed: %w", err)
	}

	if len(resp.FeatureVectors) == 0 {
		return make(map[string]float64), nil
	}

	// 转换特征向量为 map[string]float64
	features := make(map[string]float64)
	for k, v := range resp.FeatureVectors[0].Values {
		if fv, ok := convertToFloat64(v); ok {
			features[k] = fv
		}
	}

	return features, nil
}

// BatchGetUserFeatures 批量获取用户特征
func (a *FeatureServiceAdapter) BatchGetUserFeatures(ctx context.Context, userIDs []string) (map[string]map[string]float64, error) {
	if len(a.featureMapping.UserFeatures) == 0 {
		result := make(map[string]map[string]float64)
		for _, userID := range userIDs {
			result[userID] = make(map[string]float64)
		}
		return result, nil
	}

	// 构建实体行
	entityRows := make([]map[string]interface{}, len(userIDs))
	for i, userID := range userIDs {
		entityRows[i] = map[string]interface{}{
			a.featureMapping.UserEntityKey: userID,
		}
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.UserFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast batch get user features failed: %w", err)
	}

	// 转换特征向量为 map[string]map[string]float64
	result := make(map[string]map[string]float64)
	for i, fv := range resp.FeatureVectors {
		if i >= len(userIDs) {
			break
		}
		userID := userIDs[i]
		features := make(map[string]float64)
		for k, v := range fv.Values {
			if fv, ok := convertToFloat64(v); ok {
				features[k] = fv
			}
		}
		result[userID] = features
	}

	return result, nil
}

// GetItemFeatures 获取物品特征
func (a *FeatureServiceAdapter) GetItemFeatures(ctx context.Context, itemID string) (map[string]float64, error) {
	if len(a.featureMapping.ItemFeatures) == 0 {
		return make(map[string]float64), nil
	}

	// 构建实体行
	entityRows := []map[string]interface{}{
		{a.featureMapping.ItemEntityKey: itemID},
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.ItemFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast get item features failed: %w", err)
	}

	if len(resp.FeatureVectors) == 0 {
		return make(map[string]float64), nil
	}

	// 转换特征向量为 map[string]float64
	features := make(map[string]float64)
	for k, v := range resp.FeatureVectors[0].Values {
		if fv, ok := convertToFloat64(v); ok {
			features[k] = fv
		}
	}

	return features, nil
}

// BatchGetItemFeatures 批量获取物品特征
func (a *FeatureServiceAdapter) BatchGetItemFeatures(ctx context.Context, itemIDs []string) (map[string]map[string]float64, error) {
	if len(a.featureMapping.ItemFeatures) == 0 {
		result := make(map[string]map[string]float64)
		for _, itemID := range itemIDs {
			result[itemID] = make(map[string]float64)
		}
		return result, nil
	}

	// 构建实体行
	entityRows := make([]map[string]interface{}, len(itemIDs))
	for i, itemID := range itemIDs {
		entityRows[i] = map[string]interface{}{
			a.featureMapping.ItemEntityKey: itemID,
		}
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.ItemFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast batch get item features failed: %w", err)
	}

	// 转换特征向量为 map[string]map[string]float64
	result := make(map[string]map[string]float64)
	for i, fv := range resp.FeatureVectors {
		if i >= len(itemIDs) {
			break
		}
		itemID := itemIDs[i]
		features := make(map[string]float64)
		for k, v := range fv.Values {
			if fv, ok := convertToFloat64(v); ok {
				features[k] = fv
			}
		}
		result[itemID] = features
	}

	return result, nil
}

// GetRealtimeFeatures 获取实时特征
func (a *FeatureServiceAdapter) GetRealtimeFeatures(ctx context.Context, userID, itemID string) (map[string]float64, error) {
	if len(a.featureMapping.RealtimeFeatures) == 0 {
		return make(map[string]float64), nil
	}

	// 构建实体行
	entityRows := []map[string]interface{}{
		{
			a.featureMapping.UserEntityKey: userID,
			a.featureMapping.ItemEntityKey: itemID,
		},
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.RealtimeFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast get realtime features failed: %w", err)
	}

	if len(resp.FeatureVectors) == 0 {
		return make(map[string]float64), nil
	}

	// 转换特征向量为 map[string]float64
	features := make(map[string]float64)
	for k, v := range resp.FeatureVectors[0].Values {
		if fv, ok := convertToFloat64(v); ok {
			features[k] = fv
		}
	}

	return features, nil
}

// BatchGetRealtimeFeatures 批量获取实时特征
func (a *FeatureServiceAdapter) BatchGetRealtimeFeatures(ctx context.Context, pairs []core.FeatureUserItemPair) (map[core.FeatureUserItemPair]map[string]float64, error) {
	if len(a.featureMapping.RealtimeFeatures) == 0 {
		result := make(map[core.FeatureUserItemPair]map[string]float64)
		for _, pair := range pairs {
			result[pair] = make(map[string]float64)
		}
		return result, nil
	}

	// 构建实体行
	entityRows := make([]map[string]interface{}, len(pairs))
	for i, pair := range pairs {
		entityRows[i] = map[string]interface{}{
			a.featureMapping.UserEntityKey: pair.UserID,
			a.featureMapping.ItemEntityKey: pair.ItemID,
		}
	}

	// 调用 Feast 客户端
	req := &common.GetOnlineFeaturesRequest{
		Features:   a.featureMapping.RealtimeFeatures,
		EntityRows: entityRows,
	}

	resp, err := a.client.GetOnlineFeatures(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("feast batch get realtime features failed: %w", err)
	}

	// 转换特征向量为 map[FeatureUserItemPair]map[string]float64
	result := make(map[core.FeatureUserItemPair]map[string]float64)
	for i, fv := range resp.FeatureVectors {
		if i >= len(pairs) {
			break
		}
		pair := pairs[i]
		features := make(map[string]float64)
		for k, v := range fv.Values {
			if fv, ok := convertToFloat64(v); ok {
				features[k] = fv
			}
		}
		result[pair] = features
	}

	return result, nil
}

// Close 关闭特征服务（实现 core.FeatureService 接口）
func (a *FeatureServiceAdapter) Close(ctx context.Context) error {
	return a.client.Close()
}

// HTTPClient 是 Feast Feature Store 的 HTTP 客户端实现。
//
// 注意：此实现位于扩展包中，需要单独引入：
//
//	go get github.com/rushteam/reckit/ext/feast/http
type HTTPClient struct {
	// Endpoint 服务端点，例如 "http://localhost:6566"
	Endpoint string

	// Project 项目名称
	Project string

	// Timeout 超时时间
	Timeout time.Duration

	// Auth 认证信息
	Auth *common.AuthConfig

	// httpClient HTTP 客户端
	httpClient *http.Client
}

// NewClient 创建一个新的 Feast HTTP 客户端。
func NewClient(endpoint, project string, opts ...common.ClientOption) (common.Client, error) {
	config := &common.ClientConfig{
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
func (c *HTTPClient) GetOnlineFeatures(ctx context.Context, req *common.GetOnlineFeaturesRequest) (*common.GetOnlineFeaturesResponse, error) {
	// 1. 验证请求
	if len(req.Features) == 0 {
		return nil, fmt.Errorf("features are required")
	}
	if len(req.EntityRows) == 0 {
		return nil, fmt.Errorf("entity rows are required")
	}

	// 2. 构建请求体（Feast HTTP API 格式）
	body := map[string]interface{}{
		"features":           req.Features,
		"entities":           req.EntityRows,
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
	featureVectors := make([]common.FeatureVector, len(result.Results))
	for i, r := range result.Results {
		featureVectors[i] = common.FeatureVector{
			Values:    r.Values,
			EntityRow: req.EntityRows[i],
		}
	}

	return &common.GetOnlineFeaturesResponse{
		FeatureVectors: featureVectors,
		Metadata:       result.Metadata,
	}, nil
}

// GetHistoricalFeatures 获取历史特征
func (c *HTTPClient) GetHistoricalFeatures(ctx context.Context, req *common.GetHistoricalFeaturesRequest) (*common.GetHistoricalFeaturesResponse, error) {
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

	return &common.GetHistoricalFeaturesResponse{
		DataFrame: result.DataFrame,
		Metadata:  result.Metadata,
	}, nil
}

// Materialize 将特征物化到在线存储
func (c *HTTPClient) Materialize(ctx context.Context, req *common.MaterializeRequest) error {
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
func (c *HTTPClient) ListFeatures(ctx context.Context) ([]common.Feature, error) {
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
		Features []common.Feature `json:"features"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return result.Features, nil
}

// GetFeatureService 获取特征服务信息
func (c *HTTPClient) GetFeatureService(ctx context.Context) (*common.FeatureServiceInfo, error) {
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
	var info common.FeatureServiceInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return &info, nil
}

// Close 关闭连接（实现 common.Client 接口）
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

// convertToFloat64 将 interface{} 转换为 float64
func convertToFloat64(v interface{}) (float64, bool) {
	switch val := v.(type) {
	case float64:
		return val, true
	case float32:
		return float64(val), true
	case int:
		return float64(val), true
	case int64:
		return float64(val), true
	case int32:
		return float64(val), true
	default:
		return 0, false
	}
}

var (
	_ core.FeatureService = (*FeatureServiceAdapter)(nil)
	_ common.Client       = (*HTTPClient)(nil)
)
