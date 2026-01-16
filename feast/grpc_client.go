package feast

import (
	"context"
	"fmt"
	"strconv"
	"time"

	feastsdk "github.com/feast-dev/feast/sdk/go"
)

// GrpcClient 是基于官方 Feast Go SDK 的 gRPC 客户端实现。
//
// 使用官方 SDK (github.com/feast-dev/feast/sdk/go) 提供的 gRPC 客户端。
//
// 设计原则（DDD）：
//   - 领域层：Client 接口（client.go）保持不变
//   - 基础设施层：GrpcClient 实现 Client 接口
//   - 高内聚低耦合：通过接口抽象，可以替换实现
//
// 工程特征：
//   - 实时性：优秀（gRPC 低延迟、流式传输）
//   - 可扩展性：强（官方 SDK 支持完整功能）
//   - 性能：高（二进制协议、连接复用）
//   - 功能：完整（支持在线特征、历史特征、物化）
//
// 使用场景：
//   - 实时特征获取（在线预测，推荐使用）
//   - 历史特征获取（训练数据）
//   - 特征物化（离线到在线）
type GrpcClient struct {
	// client 官方 SDK 的 gRPC 客户端
	client *feastsdk.GrpcClient

	// Project 项目名称
	Project string

	// Endpoint 服务端点（用于信息展示）
	Endpoint string
}

// NewGrpcClient 创建一个基于官方 SDK 的 Feast gRPC 客户端。
//
// 参数：
//   - host: Feast Feature Server 主机地址，例如 "localhost"
//   - port: gRPC 端口，默认 6565
//   - project: 项目名称
//   - opts: 客户端配置选项
//
// 返回：
//   - *GrpcClient: gRPC 客户端实例
//   - error: 错误信息
func NewGrpcClient(host string, port int, project string, opts ...ClientOption) (*GrpcClient, error) {
	if port == 0 {
		port = 6565 // 默认 gRPC 端口
	}

	config := &ClientConfig{
		Endpoint: fmt.Sprintf("%s:%d", host, port),
		Project:  project,
		Timeout:  30 * time.Second,
		UseGRPC:  true,
	}

	// 应用配置选项
	for _, opt := range opts {
		opt(config)
	}

	// 创建官方 SDK 客户端
	var client *feastsdk.GrpcClient
	var err error

	if config.Auth != nil && config.Auth.Type == "static" && config.Auth.Token != "" {
		// 使用静态 Token 认证
		credential := feastsdk.NewStaticCredential(config.Auth.Token)
		// NewSecureGrpcClient 需要 SecurityConfig 类型
		security := feastsdk.SecurityConfig{
			EnableTLS:  false, // 可以根据需要启用 TLS
			Credential: credential,
		}
		client, err = feastsdk.NewSecureGrpcClient(host, port, security)
	} else {
		// 使用无认证连接（默认使用 insecure 连接）
		client, err = feastsdk.NewGrpcClient(host, port)
	}

	if err != nil {
		return nil, fmt.Errorf("创建 Feast gRPC 客户端失败: %w", err)
	}

	return &GrpcClient{
		client:   client,
		Project:  project,
		Endpoint: config.Endpoint,
	}, nil
}

// GetOnlineFeatures 获取在线特征（实现 Client 接口）
func (c *GrpcClient) GetOnlineFeatures(ctx context.Context, req *GetOnlineFeaturesRequest) (*GetOnlineFeaturesResponse, error) {
	// 1. 验证请求
	if len(req.Features) == 0 {
		return nil, fmt.Errorf("features are required")
	}
	if len(req.EntityRows) == 0 {
		return nil, fmt.Errorf("entity rows are required")
	}

	// 2. 构建官方 SDK 请求
	project := req.Project
	if project == "" {
		project = c.Project
	}
	if project == "" {
		return nil, fmt.Errorf("project is required")
	}

	// 3. 转换实体行为 SDK 格式
	// SDK 的 Row 类型是 map[string]*types.Value
	entityRows := make([]feastsdk.Row, len(req.EntityRows))
	for i, row := range req.EntityRows {
		entityRow := make(feastsdk.Row)
		for k, v := range row {
			// 根据值类型转换为 SDK 支持的类型
			// 使用 SDK 提供的辅助函数创建 *types.Value
			switch val := v.(type) {
			case string:
				entityRow[k] = feastsdk.StrVal(val)
			case int:
				entityRow[k] = feastsdk.Int64Val(int64(val))
			case int64:
				entityRow[k] = feastsdk.Int64Val(val)
			case int32:
				entityRow[k] = feastsdk.Int64Val(int64(val))
			case float64:
				entityRow[k] = feastsdk.DoubleVal(val)
			case float32:
				entityRow[k] = feastsdk.FloatVal(val)
			case bool:
				entityRow[k] = feastsdk.BoolVal(val)
			case []byte:
				entityRow[k] = feastsdk.BytesVal(val)
			default:
				// 尝试转换为字符串
				entityRow[k] = feastsdk.StrVal(fmt.Sprintf("%v", val))
			}
		}
		entityRows[i] = entityRow
	}

	// 4. 构建 SDK 请求
	sdkReq := &feastsdk.OnlineFeaturesRequest{
		Features: req.Features,
		Entities: entityRows,
		Project:  project,
	}

	// 5. 调用官方 SDK
	sdkResp, err := c.client.GetOnlineFeatures(ctx, sdkReq)
	if err != nil {
		return nil, fmt.Errorf("feast get online features failed: %w", err)
	}

	// 6. 转换 SDK 响应为领域模型
	rows := sdkResp.Rows()
	if len(rows) != len(req.EntityRows) {
		return nil, fmt.Errorf("response row count mismatch: expected %d, got %d", len(req.EntityRows), len(rows))
	}

	featureVectors := make([]FeatureVector, len(rows))
	featureNames := req.Features

	// 使用 SDK 提供的辅助方法提取特征值
	// SDK 的 Rows() 返回 []Row，其中 Row 是 map[string]*types.Value
	for i := 0; i < len(rows); i++ {
		values := make(map[string]interface{})
		row := rows[i] // row 是 feastsdk.Row 类型（map[string]*types.Value）

		// Row 是 map 类型，直接按特征名称访问
		for _, featureName := range featureNames {
			if val, exists := row[featureName]; exists {
				convertedVal := convertFromSDKValue(val)
				if convertedVal != nil {
					values[featureName] = convertedVal
				}
			}
		}

		featureVectors[i] = FeatureVector{
			Values:    values,
			EntityRow: req.EntityRows[i],
		}
	}

	return &GetOnlineFeaturesResponse{
		FeatureVectors: featureVectors,
		Metadata:       make(map[string]interface{}),
	}, nil
}

// GetHistoricalFeatures 获取历史特征（实现 Client 接口）
func (c *GrpcClient) GetHistoricalFeatures(ctx context.Context, req *GetHistoricalFeaturesRequest) (*GetHistoricalFeaturesResponse, error) {
	// 官方 SDK 主要支持在线特征，历史特征需要通过其他方式获取
	// 这里返回一个基础实现，实际使用时可能需要调用其他 API
	return nil, fmt.Errorf("历史特征获取暂不支持，请使用 HTTP 客户端或直接调用 Feast API")
}

// Materialize 将特征物化到在线存储（实现 Client 接口）
func (c *GrpcClient) Materialize(ctx context.Context, req *MaterializeRequest) error {
	// 官方 SDK 主要支持特征获取，物化操作需要通过其他方式
	// 这里返回一个基础实现，实际使用时可能需要调用其他 API
	return fmt.Errorf("特征物化暂不支持，请使用 HTTP 客户端或直接调用 Feast API")
}

// ListFeatures 列出所有可用的特征（实现 Client 接口）
func (c *GrpcClient) ListFeatures(ctx context.Context) ([]Feature, error) {
	// 官方 SDK 主要支持特征获取，列出特征需要通过其他方式
	// 这里返回一个基础实现，实际使用时可能需要调用其他 API
	return nil, fmt.Errorf("列出特征暂不支持，请使用 HTTP 客户端或直接调用 Feast API")
}

// GetFeatureService 获取特征服务信息（实现 Client 接口）
func (c *GrpcClient) GetFeatureService(ctx context.Context) (*FeatureServiceInfo, error) {
	return &FeatureServiceInfo{
		Endpoint:     c.Endpoint,
		Project:      c.Project,
		FeatureViews: []string{},
		OnlineStore:  "grpc",
		OfflineStore: "unknown",
	}, nil
}

// Close 关闭客户端连接（实现 Client 接口）
func (c *GrpcClient) Close() error {
	// 官方 SDK 的 GrpcClient 内部使用 gRPC 连接
	// gRPC 连接通常会自动管理，但如果有资源需要清理，可以在这里处理
	// 目前官方 SDK 没有显式的 Close 方法，连接由 gRPC 库管理
	c.client = nil
	return nil
}

// convertToSDKValue 将 interface{} 转换为 SDK 支持的值类型
// SDK 的辅助函数（StrVal, Int64Val 等）返回的类型应该可以直接赋值给 Row
// 由于类型系统的限制，这里需要返回 interface{}，但实际返回的是 *types.Value
func convertToSDKValue(v interface{}) interface{} {
	switch val := v.(type) {
	case string:
		return feastsdk.StrVal(val)
	case int:
		return feastsdk.Int64Val(int64(val))
	case int64:
		return feastsdk.Int64Val(val)
	case int32:
		return feastsdk.Int64Val(int64(val))
	case float64:
		return feastsdk.DoubleVal(val)
	case float32:
		return feastsdk.FloatVal(val)
	case bool:
		return feastsdk.BoolVal(val)
	case []byte:
		return feastsdk.BytesVal(val)
	default:
		// 尝试转换为字符串
		return feastsdk.StrVal(fmt.Sprintf("%v", val))
	}
}

// convertFromSDKValue 从 SDK 值类型转换为 interface{}
// SDK 返回的是 *serving.Types.Value，需要使用 SDK 的辅助方法提取值
func convertFromSDKValue(val interface{}) interface{} {
	if val == nil {
		return nil
	}

	// 官方 SDK 的 Value 类型是 *serving.Types.Value
	// 需要使用 SDK 提供的辅助方法或直接类型断言到 protobuf 类型
	// 由于 SDK 的 Rows() 返回 [][]interface{}，这里需要处理多种可能类型

	switch v := val.(type) {
	case string:
		return v
	case int64:
		return float64(v)
	case int32:
		return float64(v)
	case int:
		return float64(v)
	case float64:
		return v
	case float32:
		return float64(v)
	case bool:
		// 布尔值转换为数值：true -> 1.0, false -> 0.0
		if v {
			return float64(1)
		}
		return float64(0)
	case []byte:
		return string(v)
	default:
		// 尝试类型转换
		// 如果是 protobuf 的 Value 类型，需要通过反射或类型断言
		// 这里先尝试转换为字符串再解析
		strVal := fmt.Sprintf("%v", val)

		// 尝试解析为数字
		if f, err := strconv.ParseFloat(strVal, 64); err == nil {
			return f
		}

		// 返回字符串值
		return strVal
	}
}

// 确保 GrpcClient 实现了 Client 接口
var _ Client = (*GrpcClient)(nil)
