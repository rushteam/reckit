package feast

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// HTTPClient 是 Feast Feature Store 的 HTTP 客户端实现。
type HTTPClient struct {
	Endpoint   string
	Project    string
	Timeout    time.Duration
	Auth       *AuthConfig
	httpClient *http.Client
}

// NewHTTPClient 创建一个新的 Feast HTTP 客户端。
func NewHTTPClient(endpoint, project string, opts ...ClientOption) (Client, error) {
	config := &ClientConfig{
		Endpoint: endpoint,
		Project:  project,
		Timeout:  30 * time.Second,
		UseGRPC:  false,
	}
	for _, opt := range opts {
		opt(config)
	}
	return &HTTPClient{
		Endpoint:   config.Endpoint,
		Project:    config.Project,
		Timeout:    config.Timeout,
		Auth:       config.Auth,
		httpClient: &http.Client{Timeout: config.Timeout},
	}, nil
}

// GetOnlineFeatures 获取在线特征
func (c *HTTPClient) GetOnlineFeatures(ctx context.Context, req *GetOnlineFeaturesRequest) (*GetOnlineFeaturesResponse, error) {
	if len(req.Features) == 0 {
		return nil, fmt.Errorf("features are required")
	}
	if len(req.EntityRows) == 0 {
		return nil, fmt.Errorf("entity rows are required")
	}
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
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "get-online-features"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.Auth != nil {
		c.addAuth(httpReq)
	}
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("feast error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}
	var result struct {
		Metadata map[string]interface{} `json:"metadata"`
		Results  []struct {
			Values map[string]interface{} `json:"values"`
		} `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	featureVectors := make([]FeatureVector, len(result.Results))
	for i, r := range result.Results {
		featureVectors[i] = FeatureVector{Values: r.Values, EntityRow: req.EntityRows[i]}
	}
	return &GetOnlineFeaturesResponse{FeatureVectors: featureVectors, Metadata: result.Metadata}, nil
}

// GetHistoricalFeatures 获取历史特征
func (c *HTTPClient) GetHistoricalFeatures(ctx context.Context, req *GetHistoricalFeaturesRequest) (*GetHistoricalFeaturesResponse, error) {
	if len(req.Features) == 0 {
		return nil, fmt.Errorf("features are required")
	}
	if len(req.EntityDF) == 0 {
		return nil, fmt.Errorf("entity df is required")
	}
	body := map[string]interface{}{"features": req.Features, "entity_df": req.EntityDF}
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
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "get-historical-features"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.Auth != nil {
		c.addAuth(httpReq)
	}
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("feast error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}
	var result struct {
		DataFrame []map[string]interface{} `json:"dataframe"`
		Metadata  map[string]interface{}   `json:"metadata"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &GetHistoricalFeaturesResponse{DataFrame: result.DataFrame, Metadata: result.Metadata}, nil
}

// Materialize 将特征物化到在线存储
func (c *HTTPClient) Materialize(ctx context.Context, req *MaterializeRequest) error {
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
	jsonData, err := json.Marshal(body)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "materialize"
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.Auth != nil {
		c.addAuth(httpReq)
	}
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("feast error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}
	return nil
}

// ListFeatures 列出所有可用的特征
func (c *HTTPClient) ListFeatures(ctx context.Context) ([]Feature, error) {
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "list-features"
	if c.Project != "" {
		url += "?project=" + c.Project
	}
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	if c.Auth != nil {
		c.addAuth(httpReq)
	}
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("feast error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}
	var result struct {
		Features []Feature `json:"features"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return result.Features, nil
}

// GetFeatureService 获取特征服务信息
func (c *HTTPClient) GetFeatureService(ctx context.Context) (*FeatureServiceInfo, error) {
	url := c.Endpoint
	if url[len(url)-1] != '/' {
		url += "/"
	}
	url += "info"
	if c.Project != "" {
		url += "?project=" + c.Project
	}
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	if c.Auth != nil {
		c.addAuth(httpReq)
	}
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("feast error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}
	var info FeatureServiceInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return &info, nil
}

// Close 关闭连接
func (c *HTTPClient) Close() error {
	return nil
}

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

var _ Client = (*HTTPClient)(nil)
