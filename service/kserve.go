package service

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"time"

	"github.com/rushteam/reckit/core"
)

// KServeProtocol 指定 KServe 协议版本。
const (
	KServeV1 = "v1"
	KServeV2 = "v2"
)

// KServeClient 是 KServe V1/V2 协议的客户端实现，用于从 KServe 提供的推理服务获取预测结果。
//
// KServe V1（基于 TensorFlow Serving REST）：
//   - Predict: POST /v1/models/{model_name}:predict
//   - 请求：{"instances": [...]} 或 {"inputs": [...]}
//   - 响应：{"predictions": [...]}
//   - Model Ready: GET /v1/models/{model_name}
//
// KServe V2（Open Inference Protocol）：
//   - Infer: POST /v2/models/{model_name}[/versions/{version}]/infer
//   - 请求：{"inputs": [{"name": "input0", "shape": [batch, dim], "datatype": "FP32", "data": [...]}]}
//   - 响应：{"outputs": [{"name": "...", "data": [...]}]}
//   - Server Ready: GET /v2/health/ready；Model Ready: GET /v2/models/{model_name}/ready
//
// 使用场景：KServe / ModelMesh 部署的模型、兼容 KServe 协议的自建推理服务。
type KServeClient struct {
	// Endpoint 服务根地址，如 "http://localhost:8000"
	Endpoint string
	// ModelName 模型名称
	ModelName string
	// ModelVersion 模型版本（可选，V2 路径中会带 /versions/{version}）
	ModelVersion string
	// Protocol 协议版本："v1" 或 "v2"，默认 "v2"
	Protocol string
	// V2InputName V2 协议下输入张量名称，默认 "input0"
	V2InputName string
	// V2OutputName V2 协议下期望的输出张量名称，用于从 outputs 中取第一个匹配或第一个；空则取 outputs[0]
	V2OutputName string
	// Timeout 请求超时
	Timeout time.Duration
	// Auth 认证配置
	Auth *AuthConfig
	// httpClient 自定义 HTTP 客户端（可选）
	httpClient *http.Client
}

// NewKServeClient 创建 KServe 客户端。endpoint 为根地址（如 http://localhost:8000），modelName 为模型名。
func NewKServeClient(endpoint, modelName string, opts ...KServeOption) *KServeClient {
	c := &KServeClient{
		Endpoint:     endpoint,
		ModelName:    modelName,
		Protocol:     KServeV2,
		V2InputName: "input0",
		Timeout:      30 * time.Second,
	}
	for _, opt := range opts {
		opt(c)
	}
	if c.httpClient == nil {
		c.httpClient = &http.Client{Timeout: c.Timeout}
	}
	return c
}

// KServeOption 配置 KServe 客户端
type KServeOption func(*KServeClient)

// WithKServeVersion 设置模型版本（V2 路径会带 /versions/{version}）
func WithKServeVersion(version string) KServeOption {
	return func(c *KServeClient) {
		c.ModelVersion = version
	}
}

// WithKServeProtocol 设置协议："v1" 或 "v2"
func WithKServeProtocol(protocol string) KServeOption {
	return func(c *KServeClient) {
		if protocol == KServeV1 || protocol == KServeV2 {
			c.Protocol = protocol
		}
	}
}

// WithKServeV2InputName 设置 V2 协议下输入张量名称
func WithKServeV2InputName(name string) KServeOption {
	return func(c *KServeClient) {
		c.V2InputName = name
	}
}

// WithKServeV2OutputName 设置 V2 协议下期望的输出张量名称（解析响应时优先匹配）
func WithKServeV2OutputName(name string) KServeOption {
	return func(c *KServeClient) {
		c.V2OutputName = name
	}
}

// WithKServeTimeout 设置超时
func WithKServeTimeout(timeout time.Duration) KServeOption {
	return func(c *KServeClient) {
		c.Timeout = timeout
		if c.httpClient != nil {
			c.httpClient.Timeout = timeout
		}
	}
}

// WithKServeAuth 设置认证
func WithKServeAuth(auth *AuthConfig) KServeOption {
	return func(c *KServeClient) {
		c.Auth = auth
	}
}

// WithKServeHTTPClient 设置自定义 HTTP 客户端
func WithKServeHTTPClient(client *http.Client) KServeOption {
	return func(c *KServeClient) {
		c.httpClient = client
	}
}

// Predict 实现 core.MLService。
func (c *KServeClient) Predict(ctx context.Context, req *core.MLPredictRequest) (*core.MLPredictResponse, error) {
	if len(req.Instances) == 0 && len(req.Features) == 0 {
		return nil, fmt.Errorf("instances or features are required")
	}
	if c.Protocol == KServeV1 {
		return c.predictV1(ctx, req)
	}
	return c.predictV2(ctx, req)
}

// predictV1 使用 V1 协议：POST /v1/models/{model_name}:predict，请求 instances/inputs，响应 predictions。
func (c *KServeClient) predictV1(ctx context.Context, req *core.MLPredictRequest) (*core.MLPredictResponse, error) {
	url := fmt.Sprintf("%s/v1/models/%s:predict", c.Endpoint, c.ModelName)

	var body interface{}
	if len(req.Features) > 0 {
		body = map[string]interface{}{"instances": req.Features}
	} else {
		body = map[string]interface{}{"instances": req.Instances}
	}
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("kserve v1 marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("kserve v1 create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	c.addAuth(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("kserve v1 request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("kserve v1 error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("kserve v1 read response: %w", err)
	}

	predictions, err := c.parseV1Predictions(bodyBytes)
	if err != nil {
		return nil, err
	}
	predictions = c.fixPredictionsCount(predictions, req)
	return &core.MLPredictResponse{
		Predictions:  predictions,
		Outputs:      string(bodyBytes),
		ModelVersion: c.ModelVersion,
	}, nil
}

// predictV2 使用 V2 协议：POST /v2/models/{model_name}/infer，请求 inputs 张量，响应 outputs。
func (c *KServeClient) predictV2(ctx context.Context, req *core.MLPredictRequest) (*core.MLPredictResponse, error) {
	path := fmt.Sprintf("%s/v2/models/%s", c.Endpoint, c.ModelName)
	if c.ModelVersion != "" {
		path = fmt.Sprintf("%s/versions/%s", path, c.ModelVersion)
	}
	url := path + "/infer"

	// 将 Features 或 Instances 转为 V2 的 inputs[].data（展平、行优先）
	var data []float64
	var rows int
	if len(req.Features) > 0 {
		rows = len(req.Features)
		keys := c.sortedFeatureKeys(req.Features[0])
		dim := len(keys)
		data = make([]float64, 0, rows*dim)
		for _, m := range req.Features {
			for _, k := range keys {
				data = append(data, m[k])
			}
		}
	} else {
		rows = len(req.Instances)
		if rows > 0 {
			dim := len(req.Instances[0])
			data = make([]float64, 0, rows*dim)
			for _, row := range req.Instances {
				data = append(data, row...)
			}
		}
	}

	inputName := c.V2InputName
	if inputName == "" {
		inputName = "input0"
	}
	shape := []int{rows, len(data) / rows}
	if rows == 0 {
		shape = []int{0, 0}
	}

	reqBody := map[string]interface{}{
		"inputs": []map[string]interface{}{
			{
				"name":     inputName,
				"shape":    shape,
				"datatype": "FP64",
				"data":     data,
			},
		},
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("kserve v2 marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("kserve v2 create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	c.addAuth(httpReq)

	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("kserve v2 request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("kserve v2 error: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}

	bodyBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("kserve v2 read response: %w", err)
	}

	predictions, err := c.parseV2Outputs(bodyBytes)
	if err != nil {
		return nil, err
	}
	predictions = c.fixPredictionsCount(predictions, req)
	return &core.MLPredictResponse{
		Predictions:  predictions,
		Outputs:      string(bodyBytes),
		ModelVersion: c.ModelVersion,
	}, nil
}

func (c *KServeClient) sortedFeatureKeys(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func (c *KServeClient) parseV1Predictions(body []byte) ([]float64, error) {
	var out struct {
		Predictions []interface{} `json:"predictions"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("kserve v1 parse response: %w", err)
	}
	predictions := make([]float64, 0, len(out.Predictions))
	for _, v := range out.Predictions {
		f, ok := toFloat64(v)
		if ok {
			predictions = append(predictions, f)
		} else if arr, ok := v.([]interface{}); ok && len(arr) > 0 {
			// 多输出时取第一个标量
			if f, ok = toFloat64(arr[0]); ok {
				predictions = append(predictions, f)
			}
		}
	}
	return predictions, nil
}

// v2InferResponse 对应 V2 推理响应
type v2InferResponse struct {
	ModelName    string           `json:"model_name"`
	ModelVersion string           `json:"model_version"`
	Outputs      []v2OutputTensor `json:"outputs"`
}

type v2OutputTensor struct {
	Name    string        `json:"name"`
	Shape   []int         `json:"shape"`
	Datatype string       `json:"datatype"`
	Data    []interface{} `json:"data"`
}

func (c *KServeClient) parseV2Outputs(body []byte) ([]float64, error) {
	var out v2InferResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, fmt.Errorf("kserve v2 parse response: %w", err)
	}
	if len(out.Outputs) == 0 {
		return nil, fmt.Errorf("kserve v2 empty outputs")
	}
	// 若指定了 V2OutputName 则优先匹配
	var tensor *v2OutputTensor
	for i := range out.Outputs {
		if c.V2OutputName != "" && out.Outputs[i].Name == c.V2OutputName {
			tensor = &out.Outputs[i]
			break
		}
	}
	if tensor == nil {
		tensor = &out.Outputs[0]
	}
	predictions := make([]float64, 0, len(tensor.Data))
	for _, v := range tensor.Data {
		f, ok := toFloat64(v)
		if ok {
			predictions = append(predictions, f)
		}
	}
	return predictions, nil
}

func toFloat64(v interface{}) (float64, bool) {
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
		if len(val) > 0 {
			return toFloat64(val[0])
		}
		return 0, false
	default:
		return 0, false
	}
}

func (c *KServeClient) fixPredictionsCount(predictions []float64, req *core.MLPredictRequest) []float64 {
	expected := len(req.Instances)
	if expected == 0 {
		expected = len(req.Features)
	}
	if expected <= 0 || len(predictions) == expected {
		return predictions
	}
	if len(predictions) == 1 && expected > 1 {
		v := predictions[0]
		out := make([]float64, expected)
		out[0] = v
		for i := 1; i < expected; i++ {
			out[i] = v
		}
		return out
	}
	return predictions
}

// Health 实现 core.MLService。V1 使用 GET /v1/models/{model_name}，V2 使用 GET /v2/health/ready。
func (c *KServeClient) Health(ctx context.Context) error {
	var url string
	if c.Protocol == KServeV1 {
		url = fmt.Sprintf("%s/v1/models/%s", c.Endpoint, c.ModelName)
	} else {
		url = fmt.Sprintf("%s/v2/health/ready", c.Endpoint)
	}
	httpReq, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return fmt.Errorf("kserve health create request: %w", err)
	}
	c.addAuth(httpReq)
	resp, err := c.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("kserve health request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		bodyBytes, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("kserve health failed: status=%d, body=%s", resp.StatusCode, string(bodyBytes))
	}
	return nil
}

// Close 实现 core.MLService。
func (c *KServeClient) Close(ctx context.Context) error {
	return nil
}

func (c *KServeClient) addAuth(req *http.Request) {
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

var _ core.MLService = (*KServeClient)(nil)
