package feature

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// HTTPMetadataLoader HTTP 接口特征元数据加载器
type HTTPMetadataLoader struct {
	client  *http.Client
	timeout time.Duration
}

// NewHTTPMetadataLoader 创建 HTTP 接口特征元数据加载器
//
// 用法：
//
//	loader := feature.NewHTTPMetadataLoader(5 * time.Second)
//	meta, err := loader.Load(ctx, "http://api.example.com/models/v1.0.0/feature_meta")
func NewHTTPMetadataLoader(timeout time.Duration) *HTTPMetadataLoader {
	if timeout == 0 {
		timeout = 10 * time.Second
	}
	return &HTTPMetadataLoader{
		client: &http.Client{
			Timeout: timeout,
		},
		timeout: timeout,
	}
}

// NewHTTPMetadataLoaderWithClient 使用自定义 HTTP 客户端创建加载器
func NewHTTPMetadataLoaderWithClient(client *http.Client) *HTTPMetadataLoader {
	return &HTTPMetadataLoader{
		client:  client,
		timeout: client.Timeout,
	}
}

// Load 从 HTTP 接口加载特征元数据
func (l *HTTPMetadataLoader) Load(ctx context.Context, url string) (*FeatureMetadata, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("创建 HTTP 请求失败: %w", err)
	}

	resp, err := l.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP 请求失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP 请求失败: status=%d, body=%s", resp.StatusCode, string(body))
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应失败: %w", err)
	}

	var meta FeatureMetadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, fmt.Errorf("解析特征元数据失败: %w", err)
	}

	return &meta, nil
}

// HTTPScalerLoader HTTP 接口特征标准化器加载器
type HTTPScalerLoader struct {
	client  *http.Client
	timeout time.Duration
}

// NewHTTPScalerLoader 创建 HTTP 接口特征标准化器加载器
//
// 用法：
//
//	loader := feature.NewHTTPScalerLoader(5 * time.Second)
//	scaler, err := loader.Load(ctx, "http://api.example.com/models/v1.0.0/feature_scaler")
func NewHTTPScalerLoader(timeout time.Duration) *HTTPScalerLoader {
	if timeout == 0 {
		timeout = 10 * time.Second
	}
	return &HTTPScalerLoader{
		client: &http.Client{
			Timeout: timeout,
		},
		timeout: timeout,
	}
}

// NewHTTPScalerLoaderWithClient 使用自定义 HTTP 客户端创建加载器
func NewHTTPScalerLoaderWithClient(client *http.Client) *HTTPScalerLoader {
	return &HTTPScalerLoader{
		client:  client,
		timeout: client.Timeout,
	}
}

// Load 从 HTTP 接口加载特征标准化器
func (l *HTTPScalerLoader) Load(ctx context.Context, url string) (FeatureScaler, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("创建 HTTP 请求失败: %w", err)
	}

	resp, err := l.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("HTTP 请求失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP 请求失败: status=%d, body=%s", resp.StatusCode, string(body))
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应失败: %w", err)
	}

	var scaler FeatureScaler
	if err := json.Unmarshal(data, &scaler); err != nil {
		return nil, fmt.Errorf("解析特征标准化器失败: %w", err)
	}

	return scaler, nil
}
