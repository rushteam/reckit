package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"time"

	"github.com/rushteam/reckit/feature"
)

// 示例：实现 S3Client 接口（S3 兼容协议）
// S3 兼容协议支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等
// 实际使用时，可以使用任何 S3 兼容的 SDK

// AWS S3 客户端实现示例
type AWSS3Client struct {
	// 这里应该包含实际的 S3 客户端
	// 使用 github.com/aws/aws-sdk-go/service/s3
	// client *s3.S3
}

// GetObject 实现 S3Client 接口
func (c *AWSS3Client) GetObject(ctx context.Context, bucket, key string) (io.ReadCloser, error) {
	// 实际实现应该调用 AWS S3 SDK
	// result, err := c.client.GetObjectWithContext(ctx, &s3.GetObjectInput{
	//     Bucket: aws.String(bucket),
	//     Key:    aws.String(key),
	// })
	// if err != nil {
	//     return nil, err
	// }
	// return result.Body, nil
	
	// 这里只是示例，实际需要实现
	return nil, fmt.Errorf("需要实现 S3 SDK 调用")
}

// AliyunOSSClient 阿里云 OSS 客户端实现示例（使用 S3 兼容协议）
// 阿里云 OSS 支持 S3 兼容协议，可以使用 AWS S3 SDK
type AliyunOSSClient struct {
	// 使用 AWS S3 SDK，配置 OSS 的 S3 兼容端点
	// client *s3.S3
}

// GetObject 实现 S3Client 接口
func (c *AliyunOSSClient) GetObject(ctx context.Context, bucket, key string) (io.ReadCloser, error) {
	// 使用 AWS S3 SDK，配置 endpoint 为 OSS 的 S3 兼容端点
	// 例如：oss-cn-hangzhou.aliyuncs.com
	// 这样可以使用统一的 S3 兼容协议访问 OSS
	
	// 这里只是示例，实际需要实现
	return nil, fmt.Errorf("需要实现 S3 兼容协议调用")
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	fmt.Println("=== 特征元数据加载器使用示例 ===")
	fmt.Println()

	// ========== 方式 1：本地文件加载 ==========
	fmt.Println("1. 本地文件加载")
	fileMetaLoader := feature.NewFileMetadataLoader()
	fileScalerLoader := feature.NewFileScalerLoader()

	meta, err := fileMetaLoader.Load(ctx, "../../python/model/feature_meta.json")
	if err != nil {
		log.Printf("加载失败: %v", err)
	} else {
		fmt.Printf("  ✅ 加载成功: 模型版本=%s, 特征数=%d\n", meta.ModelVersion, meta.FeatureCount)
	}

	scaler, err := fileScalerLoader.Load(ctx, "../../python/model/feature_scaler.json")
	if err != nil {
		log.Printf("加载标准化器失败: %v", err)
	} else {
		fmt.Printf("  ✅ 标准化器加载成功: 特征数=%d\n", len(scaler))
	}
	fmt.Println()

	// ========== 方式 2：HTTP 接口加载 ==========
	fmt.Println("2. HTTP 接口加载")
	// httpMetaLoader := feature.NewHTTPMetadataLoader(5 * time.Second)
	// httpScalerLoader := feature.NewHTTPScalerLoader(5 * time.Second)

	// 示例 URL（需要实际的服务地址）
	metaURL := "http://api.example.com/models/v1.0.0/feature_meta"
	scalerURL := "http://api.example.com/models/v1.0.0/feature_scaler"

	// 实际使用时取消注释
	// metaHTTP, err := httpMetaLoader.Load(ctx, metaURL)
	// if err != nil {
	//     log.Printf("从 HTTP 加载失败: %v", err)
	// } else {
	//     fmt.Printf("  ✅ 从 HTTP 加载成功: %s\n", metaHTTP.ModelVersion)
	// }

	fmt.Printf("  📝 示例 URL: %s\n", metaURL)
	fmt.Printf("  📝 示例 URL: %s\n", scalerURL)
	fmt.Println("  （需要实际的服务地址）")
	fmt.Println()

	// ========== 方式 3：S3 兼容协议加载 ==========
	fmt.Println("3. S3 兼容协议加载")
	// S3 兼容协议支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等
	// 需要实现 S3Client 接口
	// s3Client := &AWSS3Client{} // 或 AliyunOSSClient、TencentCOSClient 等
	// s3MetaLoader := feature.NewS3MetadataLoader(s3Client, "my-model-bucket")
	// s3ScalerLoader := feature.NewS3ScalerLoader(s3Client, "my-model-bucket")

	// 示例 key
	metaKey := "models/v1.0.0/feature_meta.json"
	scalerKey := "models/v1.0.0/feature_scaler.json"

	// 实际使用时取消注释
	// metaS3, err := s3MetaLoader.Load(ctx, metaKey)
	// if err != nil {
	//     log.Printf("从 S3 兼容存储加载失败: %v", err)
	// } else {
	//     fmt.Printf("  ✅ 从 S3 兼容存储加载成功: %s\n", metaS3.ModelVersion)
	// }

	fmt.Printf("  📝 示例 Bucket: my-model-bucket\n")
	fmt.Printf("  📝 示例 Key: %s\n", metaKey)
	fmt.Printf("  📝 示例 Key: %s\n", scalerKey)
	fmt.Println("  （需要实现 S3Client 接口）")
	fmt.Println("  （支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等）")
	fmt.Println()

	// ========== 使用加载的特征元数据 ==========
	if meta != nil {
		fmt.Println("=== 使用特征元数据 ===")
		
		// 示例特征
		features := map[string]float64{
			"item_ctr":  0.15,
			"item_cvr":  0.08,
			"user_age":  25.0,
			"user_gender": 1.0,
		}

		// 验证特征
		validated := meta.ValidateFeatures(features)
		fmt.Printf("验证后的特征: %v\n", validated)

		// 检查缺失特征
		missing := meta.GetMissingFeatures(features)
		if len(missing) > 0 {
			fmt.Printf("缺失特征: %v\n", missing)
		}

		// 构建特征向量
		vector := meta.BuildFeatureVector(features)
		fmt.Printf("特征向量长度: %d\n", len(vector))

		// 标准化（如果配置了）
		if scaler != nil && meta.Normalized {
			normalized := scaler.Normalize(validated)
			fmt.Printf("标准化后的特征: %v\n", normalized)
		}
	}
}
