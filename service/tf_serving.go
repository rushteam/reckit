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

// TFServingClient 是 TensorFlow Serving 的客户端实现。
//
// TensorFlow Serving 支持两种协议：
//   - gRPC：端口 8500（推荐，性能更好）
//   - REST API：端口 8501（HTTP/JSON）
//
// 工程特征：
//   - 实时性：好（gRPC 低延迟）
//   - 可扩展性：强（支持多模型、多版本）
//   - 性能：高（gRPC 二进制协议）
//   - 功能：完整（支持模型管理、版本控制）
//
// 使用场景：
//   - TensorFlow 模型推理
//   - 大规模模型服务
//   - 需要版本管理的场景
type TFServingClient struct {
	// Endpoint 服务端点
	// gRPC: "localhost:8500"
	// REST: "http://localhost:8501"
	Endpoint string

	// ModelName 模型名称
	ModelName string

	// ModelVersion 模型版本（可选，为空则使用最新版本）
	ModelVersion string

	// SignatureName 签名名称（可选，默认为 "serving_default"）
	SignatureName string

	// Timeout 超时时间
	Timeout time.Duration

	// UseGRPC 是否使用 gRPC（默认 false，使用 REST API）
	UseGRPC bool

	// Auth 认证信息
	Auth *AuthConfig

	// httpClient HTTP 客户端（REST API 使用）
	httpClient *http.Client
}

// NewTFServingClient 创建一个新的 TF Serving 客户端。
func NewTFServingClient(endpoint, modelName string, opts ...TFServingOption) *TFServingClient {
	client := &TFServingClient{
		Endpoint:      endpoint,
		ModelName:     modelName,
		SignatureName: "serving_default",
		Timeout:       30 * time.Second,
		UseGRPC:       false, // 默认使用 REST API
	}

	for _, opt := range opts {
		opt(client)
	}

	// 创建 HTTP 客户端（REST API）
	// 注意：gRPC 需要 protobuf 依赖，这里只实现 REST API
	if !client.UseGRPC {
		client.httpClient = &http.Client{
			Timeout: client.Timeout,
		}
	} else {
		// gRPC 模式暂不支持，回退到 REST API
		client.UseGRPC = false
		client.httpClient = &http.Client{
			Timeout: client.Timeout,
		}
	}

	return client
}

// TFServingOption TF Serving 客户端配置选项
type TFServingOption func(*TFServingClient)

// WithTFServingVersion 设置模型版本
func WithTFServingVersion(version string) TFServingOption {
	return func(c *TFServingClient) {
		c.ModelVersion = version
	}
}

// WithTFServingSignature 设置签名名称
func WithTFServingSignature(signatureName string) TFServingOption {
	return func(c *TFServingClient) {
		c.SignatureName = signatureName
	}
}

// WithTFServingTimeout 设置超时时间
func WithTFServingTimeout(timeout time.Duration) TFServingOption {
	return func(c *TFServingClient) {
		c.Timeout = timeout
	}
}

// WithTFServingGRPC 使用 gRPC 协议
func WithTFServingGRPC() TFServingOption {
	return func(c *TFServingClient) {
		c.UseGRPC = true
	}
}

// WithTFServingAuth 设置认证信息
func WithTFServingAuth(auth *AuthConfig) TFServingOption {
	return func(c *TFServingClient) {
		c.Auth = auth
	}
}

// Predict 实现 core.MLService 接口
func (c *TFServingClient) Predict(ctx context.Context, req *core.MLPredictRequest) (*core.MLPredictResponse, error) {
	// 1. 验证请求
	if len(req.Instances) == 0 && len(req.Features) == 0 {
		return nil, fmt.Errorf("instances or features are required")
	}

	// 2. 使用 REST API 进行预测
	return c.predictREST(ctx, req)
}

// predictREST 使用 REST API 进行预测
func (c *TFServingClient) predictREST(ctx context.Context, req *core.MLPredictRequest) (*core.MLPredictResponse, error) {
	// 1. 构建 URL
	url := fmt.Sprintf("%s/v1/models/%s:predict", c.Endpoint, c.ModelName)
	if c.ModelVersion != "" {
		url = fmt.Sprintf("%s/v1/models/%s/versions/%s:predict", c.Endpoint, c.ModelName, c.ModelVersion)
	}

	// 2. 构建请求体
	body := make(map[string]interface{})
	if len(req.Instances) > 0 {
		body["instances"] = req.Instances
	} else if len(req.Features) > 0 {
		// TF Serving 也支持 features 格式
		body["inputs"] = req.Features
	}
	if c.SignatureName != "" {
		body["signature_name"] = c.SignatureName
	}

	// 3. 序列化请求体
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	// 4. 创建 HTTP 请求
	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	// 5. 添加认证
	if c.Auth != nil {
		c.addAuth(httpReq)
	}

	// 6. 发送请求
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	defer resp.Body.Close()

	// 7. 检查状态码
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("tf serving error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	// 8. 解析响应
	var result struct {
		Predictions []interface{} `json:"predictions"`
		Outputs     interface{}   `json:"outputs,omitempty"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	// 9. 转换预测结果
	predictions := make([]float64, 0, len(result.Predictions))
	for _, pred := range result.Predictions {
		switch v := pred.(type) {
		case float64:
			predictions = append(predictions, v)
		case []interface{}:
			// 如果返回的是数组，取第一个元素
			if len(v) > 0 {
				if fv, ok := v[0].(float64); ok {
					predictions = append(predictions, fv)
				}
			}
		case []float64:
			// 如果返回的是 float64 数组，取第一个元素
			if len(v) > 0 {
				predictions = append(predictions, v[0])
			}
		default:
			return nil, fmt.Errorf("unexpected prediction type: %T", pred)
		}
	}

	return &core.MLPredictResponse{
		Predictions:  predictions,
		Outputs:      result.Outputs,
		ModelVersion: c.ModelVersion,
	}, nil
}

// addAuth 添加认证信息到 HTTP 请求
func (c *TFServingClient) addAuth(req *http.Request) {
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

// Health 健康检查
func (c *TFServingClient) Health(ctx context.Context) error {
	// 构建 URL
	url := fmt.Sprintf("%s/v1/models/%s", c.Endpoint, c.ModelName)
	if c.ModelVersion != "" {
		url = fmt.Sprintf("%s/v1/models/%s/versions/%s", c.Endpoint, c.ModelName, c.ModelVersion)
	}

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

// Close 关闭连接
func (c *TFServingClient) Close() error {
	// HTTP 客户端不需要显式关闭
	// 如果需要，可以在这里关闭连接池等资源
	return nil
}

// 确保 TFServingClient 实现了 core.MLService 接口
var _ core.MLService = (*TFServingClient)(nil)
