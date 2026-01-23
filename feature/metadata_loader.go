package feature

import (
	"context"
)

// MetadataLoader 特征元数据加载器接口
// 支持从不同来源加载特征元数据（本地文件、HTTP 接口、S3 兼容存储等）
type MetadataLoader interface {
	// Load 加载特征元数据
	// source 是数据源标识（文件路径、URL、S3 key 等）
	Load(ctx context.Context, source string) (*FeatureMetadata, error)
}

// ScalerLoader 特征标准化器加载器接口
// 支持从不同来源加载特征标准化器（本地文件、HTTP 接口、S3 兼容存储等）
type ScalerLoader interface {
	// Load 加载特征标准化器
	// source 是数据源标识（文件路径、URL、S3 key 等）
	Load(ctx context.Context, source string) (FeatureScaler, error)
}

// FileMetadataLoader 本地文件特征元数据加载器
type FileMetadataLoader struct{}

// NewFileMetadataLoader 创建本地文件特征元数据加载器
func NewFileMetadataLoader() *FileMetadataLoader {
	return &FileMetadataLoader{}
}

// Load 从本地文件加载特征元数据
func (l *FileMetadataLoader) Load(ctx context.Context, filePath string) (*FeatureMetadata, error) {
	return LoadFeatureMetadataFromFile(filePath)
}

// FileScalerLoader 本地文件特征标准化器加载器
type FileScalerLoader struct{}

// NewFileScalerLoader 创建本地文件特征标准化器加载器
func NewFileScalerLoader() *FileScalerLoader {
	return &FileScalerLoader{}
}

// Load 从本地文件加载特征标准化器
func (l *FileScalerLoader) Load(ctx context.Context, filePath string) (FeatureScaler, error) {
	return LoadFeatureScalerFromFile(filePath)
}
