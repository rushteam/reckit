package feature

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
)

// S3Client S3 兼容协议客户端接口（不直接依赖具体 SDK，支持依赖注入）
// S3 兼容协议支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等
type S3Client interface {
	// GetObject 获取对象内容
	// bucket 是存储桶名称
	// key 是对象键（文件路径）
	GetObject(ctx context.Context, bucket, key string) (io.ReadCloser, error)
}

// S3MetadataLoader S3 兼容协议特征元数据加载器
// 支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等所有 S3 兼容的对象存储服务
type S3MetadataLoader struct {
	client S3Client
	bucket string
}

// NewS3MetadataLoader 创建 S3 兼容协议特征元数据加载器
//
// 用法：
//
//	// 需要实现 S3Client 接口（支持 AWS S3、阿里云 OSS、腾讯云 COS 等）
//	s3Client := &MyS3Client{...}
//	loader := feature.NewS3MetadataLoader(s3Client, "my-bucket")
//	meta, err := loader.Load(ctx, "models/v1.0.0/feature_meta.json")
func NewS3MetadataLoader(client S3Client, bucket string) *S3MetadataLoader {
	return &S3MetadataLoader{
		client: client,
		bucket: bucket,
	}
}

// Load 从 S3 兼容存储加载特征元数据
func (l *S3MetadataLoader) Load(ctx context.Context, key string) (*FeatureMetadata, error) {
	if l.client == nil {
		return nil, fmt.Errorf("S3 客户端未设置")
	}

	reader, err := l.client.GetObject(ctx, l.bucket, key)
	if err != nil {
		return nil, fmt.Errorf("从 S3 兼容存储获取对象失败: %w", err)
	}
	defer reader.Close()

	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("读取 S3 兼容存储对象失败: %w", err)
	}

	var meta FeatureMetadata
	if err := json.Unmarshal(data, &meta); err != nil {
		return nil, fmt.Errorf("解析特征元数据失败: %w", err)
	}

	return &meta, nil
}

// S3ScalerLoader S3 兼容协议特征标准化器加载器
// 支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等所有 S3 兼容的对象存储服务
type S3ScalerLoader struct {
	client S3Client
	bucket string
}

// NewS3ScalerLoader 创建 S3 兼容协议特征标准化器加载器
//
// 用法：
//
//	// 需要实现 S3Client 接口（支持 AWS S3、阿里云 OSS、腾讯云 COS 等）
//	s3Client := &MyS3Client{...}
//	loader := feature.NewS3ScalerLoader(s3Client, "my-bucket")
//	scaler, err := loader.Load(ctx, "models/v1.0.0/feature_scaler.json")
func NewS3ScalerLoader(client S3Client, bucket string) *S3ScalerLoader {
	return &S3ScalerLoader{
		client: client,
		bucket: bucket,
	}
}

// Load 从 S3 兼容存储加载特征标准化器
func (l *S3ScalerLoader) Load(ctx context.Context, key string) (FeatureScaler, error) {
	if l.client == nil {
		return nil, fmt.Errorf("S3 客户端未设置")
	}

	reader, err := l.client.GetObject(ctx, l.bucket, key)
	if err != nil {
		return nil, fmt.Errorf("从 S3 兼容存储获取对象失败: %w", err)
	}
	defer reader.Close()

	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("读取 S3 兼容存储对象失败: %w", err)
	}

	var scaler FeatureScaler
	if err := json.Unmarshal(data, &scaler); err != nil {
		return nil, fmt.Errorf("解析特征标准化器失败: %w", err)
	}

	return scaler, nil
}

// OSSMetadataLoader 已废弃，请使用 S3MetadataLoader
// S3 兼容协议可以同时支持 AWS S3、阿里云 OSS、腾讯云 COS 等
//
// Deprecated: 使用 NewS3MetadataLoader 代替
func NewOSSMetadataLoader(client S3Client, bucket string) *S3MetadataLoader {
	return NewS3MetadataLoader(client, bucket)
}

// OSSScalerLoader 已废弃，请使用 S3ScalerLoader
// S3 兼容协议可以同时支持 AWS S3、阿里云 OSS、腾讯云 COS 等
//
// Deprecated: 使用 NewS3ScalerLoader 代替
func NewOSSScalerLoader(client S3Client, bucket string) *S3ScalerLoader {
	return NewS3ScalerLoader(client, bucket)
}
