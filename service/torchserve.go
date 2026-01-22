package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// TorchServeClient 是 TorchServe 的客户端实现。
//
// TorchServe 支持两种协议：
//   - REST API：端口 8080（推理）、8081（管理）
//   - gRPC API：端口 7070（需要额外配置）
//
// 工程特征：
//   - 实时性：好（REST API 低延迟）
//   - 可扩展性：强（支持多模型、版本控制）
//   - 性能：高（支持批量推理）
//   - 功能：完整（支持模型管理、A/B 测试）
//
// 使用场景：
//   - PyTorch 模型推理
//   - TorchScript 模型推理
//   - 大规模模型服务
//
// REST API 格式：
//   - 推理端点：POST /predictions/{model_name}
//   - 请求体：JSON 对象或数组（根据模型 Handler 定义）
//   - 响应：直接返回预测结果（格式由模型 Handler 决定）
type TorchServeClient struct {
	// Endpoint 服务端点
	// REST: "http://localhost:8080"
	Endpoint string

	// ModelName 模型名称
	ModelName string

	// ModelVersion 模型版本（可选，TorchServe 通过模型版本管理）
	ModelVersion string

	// Timeout 超时时间
	Timeout time.Duration

	// Auth 认证信息
	Auth *AuthConfig

	// httpClient HTTP 客户端
	httpClient *http.Client
}

// NewTorchServeClient 创建一个新的 TorchServe 客户端。
func NewTorchServeClient(endpoint, modelName string, opts ...TorchServeOption) *TorchServeClient {
	client := &TorchServeClient{
		Endpoint:  endpoint,
		ModelName: modelName,
		Timeout:   30 * time.Second,
	}

	for _, opt := range opts {
		opt(client)
	}

	// 创建 HTTP 客户端
	if client.httpClient == nil {
		client.httpClient = &http.Client{
			Timeout: client.Timeout,
		}
	}

	return client
}

// TorchServeOption TorchServe 客户端配置选项
type TorchServeOption func(*TorchServeClient)

// WithTorchServeVersion 设置模型版本
func WithTorchServeVersion(version string) TorchServeOption {
	return func(c *TorchServeClient) {
		c.ModelVersion = version
	}
}

// WithTorchServeTimeout 设置超时时间
func WithTorchServeTimeout(timeout time.Duration) TorchServeOption {
	return func(c *TorchServeClient) {
		c.Timeout = timeout
		if c.httpClient != nil {
			c.httpClient.Timeout = timeout
		}
	}
}

// WithTorchServeAuth 设置认证信息
func WithTorchServeAuth(auth *AuthConfig) TorchServeOption {
	return func(c *TorchServeClient) {
		c.Auth = auth
	}
}

// WithTorchServeHTTPClient 设置自定义 HTTP 客户端
func WithTorchServeHTTPClient(httpClient *http.Client) TorchServeOption {
	return func(c *TorchServeClient) {
		c.httpClient = httpClient
	}
}

// Predict 批量预测
func (c *TorchServeClient) Predict(ctx context.Context, req *PredictRequest) (*PredictResponse, error) {
	// 1. 验证请求
	if len(req.Instances) == 0 && len(req.Features) == 0 {
		return nil, fmt.Errorf("instances or features are required")
	}

	// 2. 使用 REST API 进行预测
	return c.predictREST(ctx, req)
}

// predictREST 使用 REST API 进行预测
func (c *TorchServeClient) predictREST(ctx context.Context, req *PredictRequest) (*PredictResponse, error) {
	// 1. 构建 URL
	// TorchServe 推理端点格式：/predictions/{model_name}
	url := fmt.Sprintf("%s/predictions/%s", c.Endpoint, c.ModelName)
	if c.ModelVersion != "" {
		// 如果指定了版本，可以通过查询参数传递（TorchServe 支持）
		url = fmt.Sprintf("%s?version=%s", url, c.ModelVersion)
	}

	// 2. 构建请求体
	// TorchServe 的请求体格式取决于模型的 Handler
	// 常见格式：
	//   - 单个样本：{"data": {...}}
	//   - 批量样本：{"data": [{...}, {...}]}
	//   - 或者直接发送数组：[{...}, {...}]
	var requestBody interface{}

	if len(req.Instances) > 0 {
		// 如果有 Instances（特征向量数组），转换为 TorchServe 格式
		// 假设模型期望的格式是 {"data": [[f1, f2, ...], [f1, f2, ...]]}
		requestBody = map[string]interface{}{
			"data": req.Instances,
		}
	} else if len(req.Features) > 0 {
		// 如果有 Features（特征字典数组），转换为 TorchServe 格式
		// 假设模型期望的格式是 {"data": [{...}, {...}]}
		requestBody = map[string]interface{}{
			"data": req.Features,
		}
	}

	// 3. 序列化请求体
	jsonData, err := json.Marshal(requestBody)
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
		return nil, fmt.Errorf("torchserve request failed: %w", err)
	}
	defer resp.Body.Close()

	// 7. 检查状态码
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("torchserve error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	// 8. 解析响应
	// TorchServe 的响应格式取决于模型的 Handler
	// 常见格式：
	//   - 单个预测：{"prediction": 0.85}
	//   - 批量预测：[0.85, 0.72, ...]
	//   - 或者直接返回数组：[0.85, 0.72, ...]
	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	// 尝试解析为不同的格式
	var predictions []float64

	// 尝试解析为数组格式
	var arrayResult []interface{}
	if err := json.Unmarshal(bodyBytes, &arrayResult); err == nil {
		// 成功解析为数组
		predictions = make([]float64, 0, len(arrayResult))
		for _, item := range arrayResult {
			if fv, ok := c.toFloat64(item); ok {
				predictions = append(predictions, fv)
			}
		}
	} else {
		// 尝试解析为对象格式 {"prediction": ...} 或 {"predictions": [...]}
		var objResult map[string]interface{}
		if err := json.Unmarshal(bodyBytes, &objResult); err == nil {
			// 查找常见的键名
			if pred, ok := objResult["prediction"]; ok {
				if fv, ok := c.toFloat64(pred); ok {
					predictions = []float64{fv}
				}
			} else if preds, ok := objResult["predictions"]; ok {
				// 批量预测
				if arr, ok := preds.([]interface{}); ok {
					predictions = make([]float64, 0, len(arr))
					for _, item := range arr {
						if fv, ok := c.toFloat64(item); ok {
							predictions = append(predictions, fv)
						}
					}
				}
			} else {
				// 尝试直接提取数值
				for _, v := range objResult {
					if fv, ok := c.toFloat64(v); ok {
						predictions = append(predictions, fv)
						break
					}
				}
			}
		} else {
			// 尝试直接解析为单个数值
			var singleValue float64
			if err := json.Unmarshal(bodyBytes, &singleValue); err == nil {
				predictions = []float64{singleValue}
			} else {
				return nil, fmt.Errorf("unable to parse response: %s", string(bodyBytes))
			}
		}
	}

	// 验证预测结果数量
	expectedCount := len(req.Instances)
	if expectedCount == 0 {
		expectedCount = len(req.Features)
	}
	if expectedCount > 0 && len(predictions) != expectedCount {
		// 如果只有一个预测结果但请求了多个，则复制结果
		if len(predictions) == 1 && expectedCount > 1 {
			result := predictions[0]
			predictions = make([]float64, expectedCount)
			for i := range predictions {
				predictions[i] = result
			}
		} else if len(predictions) == 0 {
			return nil, fmt.Errorf("empty predictions in response")
		}
	}

	return &PredictResponse{
		Predictions:  predictions,
		Outputs:      string(bodyBytes), // 保存原始响应用于调试
		ModelVersion: c.ModelVersion,
	}, nil
}

// toFloat64 将值转换为 float64
func (c *TorchServeClient) toFloat64(v interface{}) (float64, bool) {
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
	case []interface{}:
		// 如果是数组，取第一个元素
		if len(val) > 0 {
			return c.toFloat64(val[0])
		}
		return 0, false
	default:
		return 0, false
	}
}

// addAuth 添加认证信息到 HTTP 请求
func (c *TorchServeClient) addAuth(req *http.Request) {
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
func (c *TorchServeClient) Health(ctx context.Context) error {
	// TorchServe 健康检查端点：/ping
	url := fmt.Sprintf("%s/ping", c.Endpoint)

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
func (c *TorchServeClient) Close() error {
	// HTTP 客户端不需要显式关闭
	// 如果需要，可以在这里关闭连接池等资源
	return nil
}

// 确保 TorchServeClient 实现了 MLService 接口
var _ MLService = (*TorchServeClient)(nil)
