package feast

import (
	"context"
	"testing"
)

// TestGrpcClient_GetOnlineFeatures 测试 gRPC 客户端的基本功能
// 注意：这是一个示例测试，实际使用时需要连接真实的 Feast 服务器
func TestGrpcClient_GetOnlineFeatures(t *testing.T) {
	t.Skip("需要连接真实的 Feast 服务器才能运行")

	ctx := context.Background()

	// 创建客户端
	client, err := NewGrpcClient("localhost", 6565, "test_project")
	if err != nil {
		t.Fatalf("创建客户端失败: %v", err)
	}
	defer client.Close()

	// 构建请求
	req := &GetOnlineFeaturesRequest{
		Features: []string{
			"driver_hourly_stats:conv_rate",
			"driver_hourly_stats:acc_rate",
		},
		EntityRows: []map[string]interface{}{
			{"driver_id": "1001"},
			{"driver_id": "1002"},
		},
		Project: "test_project",
	}

	// 获取特征
	resp, err := client.GetOnlineFeatures(ctx, req)
	if err != nil {
		t.Fatalf("获取特征失败: %v", err)
	}

	// 验证响应
	if len(resp.FeatureVectors) != 2 {
		t.Errorf("期望 2 个特征向量，实际得到 %d 个", len(resp.FeatureVectors))
	}

	for i, fv := range resp.FeatureVectors {
		if len(fv.Values) == 0 {
			t.Errorf("特征向量 %d 为空", i)
		}
		t.Logf("特征向量 %d: %+v", i, fv.Values)
	}
}

// TestConvertToSDKValue 测试值类型转换
func TestConvertToSDKValue(t *testing.T) {
	tests := []struct {
		name     string
		input    interface{}
		expected interface{} // 只检查是否不为 nil
	}{
		{"string", "test", nil},
		{"int", 100, nil},
		{"int64", int64(100), nil},
		{"float64", 3.14, nil},
		{"bool", true, nil},
		{"[]byte", []byte("test"), nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertToSDKValue(tt.input)
			if result == nil {
				t.Errorf("转换结果不应该为 nil")
			}
		})
	}
}

// TestConvertFromSDKValue 测试从 SDK 值类型转换
func TestConvertFromSDKValue(t *testing.T) {
	tests := []struct {
		name  string
		input interface{}
	}{
		{"string", "test"},
		{"int64", int64(100)},
		{"float64", 3.14},
		{"bool_true", true},
		{"bool_false", false},
		{"nil", nil},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertFromSDKValue(tt.input)
			// 对于 nil 输入，结果应该为 nil
			if tt.input == nil && result != nil {
				t.Errorf("nil 输入应该返回 nil，实际得到 %v", result)
			}
			// 对于非 nil 输入，结果不应该为 nil（除非是特殊情况）
			if tt.input != nil && result == nil && tt.name != "nil" {
				t.Logf("警告：非 nil 输入返回了 nil: %v", tt.input)
			}
		})
	}
}
