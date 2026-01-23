# 特征元数据加载器示例

本示例演示如何使用不同的加载器接口加载特征元数据和标准化器。

## 支持的加载方式

### 1. 本地文件加载

适用于开发环境和本地部署。

```go
fileMetaLoader := feature.NewFileMetadataLoader()
meta, err := fileMetaLoader.Load(ctx, "python/model/feature_meta.json")
```

### 2. HTTP 接口加载

适用于生产环境，从 API 服务加载。

```go
httpMetaLoader := feature.NewHTTPMetadataLoader(5 * time.Second)
meta, err := httpMetaLoader.Load(ctx, "http://api.example.com/models/v1.0.0/feature_meta")
```

### 3. S3 兼容协议加载（推荐）

适用于云环境，从对象存储加载。**S3 兼容协议支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等**，使用统一的接口，兼容性更好。

```go
// 需要实现 S3Client 接口（S3 兼容协议）
s3Client := &MyS3Client{}
s3MetaLoader := feature.NewS3MetadataLoader(s3Client, "my-bucket")
meta, err := s3MetaLoader.Load(ctx, "models/v1.0.0/feature_meta.json")
```

**为什么使用 S3 兼容协议？**
- ✅ **统一接口**：一套代码支持多种对象存储服务
- ✅ **兼容性好**：AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等都支持 S3 兼容协议
- ✅ **易于迁移**：可以在不同云服务商之间轻松切换
- ✅ **生态丰富**：S3 兼容协议的 SDK 和工具更丰富

## S3 兼容协议客户端实现示例

### AWS S3（原生）

```go
import (
    "context"
    "io"
    "github.com/aliyun/aliyun-oss-go-sdk/oss"
)

type AliyunOSSClient struct {
    client *oss.Client
}

func NewAliyunOSSClient(endpoint, accessKeyID, accessKeySecret string) (*AliyunOSSClient, error) {
    client, err := oss.New(endpoint, accessKeyID, accessKeySecret)
    if err != nil {
        return nil, err
    }
    return &AliyunOSSClient{client: client}, nil
}

func (c *AliyunOSSClient) GetObject(ctx context.Context, bucket, key string) (io.ReadCloser, error) {
    bucketObj, err := c.client.Bucket(bucket)
    if err != nil {
        return nil, err
    }
    return bucketObj.GetObject(key)
}
```

```go
import (
    "context"
    "io"
    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/s3"
)

type AWSS3Client struct {
    client *s3.S3
}

func NewAWSS3Client(region string) (*AWSS3Client, error) {
    sess, err := session.NewSession(&aws.Config{
        Region: aws.String(region),
    })
    if err != nil {
        return nil, err
    }
    return &AWSS3Client{client: s3.New(sess)}, nil
}

func (c *AWSS3Client) GetObject(ctx context.Context, bucket, key string) (io.ReadCloser, error) {
    result, err := c.client.GetObjectWithContext(ctx, &s3.GetObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return nil, err
    }
    return result.Body, nil
}
```

### 阿里云 OSS（使用 S3 兼容协议）

阿里云 OSS 支持 S3 兼容协议，可以使用 AWS S3 SDK，只需配置 OSS 的 S3 兼容端点：

```go
import (
    "context"
    "io"
    "github.com/aws/aws-sdk-go/aws"
    "github.com/aws/aws-sdk-go/aws/credentials"
    "github.com/aws/aws-sdk-go/aws/session"
    "github.com/aws/aws-sdk-go/service/s3"
)

type AliyunOSSClient struct {
    client *s3.S3
}

// NewAliyunOSSClient 使用 S3 兼容协议访问阿里云 OSS
func NewAliyunOSSClient(endpoint, accessKeyID, accessKeySecret string) (*AliyunOSSClient, error) {
    // 配置 S3 兼容端点（例如：oss-cn-hangzhou.aliyuncs.com）
    sess, err := session.NewSession(&aws.Config{
        Region:           aws.String("cn-hangzhou"), // 区域
        Endpoint:         aws.String(endpoint),      // OSS S3 兼容端点
        S3ForcePathStyle: aws.Bool(true),            // 使用路径风格
        Credentials: credentials.NewStaticCredentials(accessKeyID, accessKeySecret, ""),
    })
    if err != nil {
        return nil, err
    }
    return &AliyunOSSClient{client: s3.New(sess)}, nil
}

func (c *AliyunOSSClient) GetObject(ctx context.Context, bucket, key string) (io.ReadCloser, error) {
    result, err := c.client.GetObjectWithContext(ctx, &s3.GetObjectInput{
        Bucket: aws.String(bucket),
        Key:    aws.String(key),
    })
    if err != nil {
        return nil, err
    }
    return result.Body, nil
}
```

### 腾讯云 COS（使用 S3 兼容协议）

```go
import (
    "context"
    "io"
    "github.com/tencentyun/cos-go-sdk-v5"
)

type COSClient struct {
    client *cos.Client
}

func NewCOSClient(secretID, secretKey, region string) (*COSClient, error) {
    u := fmt.Sprintf("https://%s.cos.%s.myqcloud.com", bucket, region)
    b := &cos.BaseURL{BucketURL: cos.NewBucketURL(u, region)}
    client := cos.NewClient(b, &http.Client{
        Transport: &cos.AuthorizationTransport{
            SecretID:  secretID,
            SecretKey: secretKey,
        },
    })
    return &COSClient{client: client}, nil
}

func (c *COSClient) GetObject(ctx context.Context, bucket, key string) (io.ReadCloser, error) {
    resp, err := c.client.Object.Get(ctx, key, nil)
    if err != nil {
        return nil, err
    }
    return resp.Body, nil
}
```

## 使用示例

```go
package main

import (
    "context"
    "time"
    "github.com/rushteam/reckit/feature"
)

func main() {
    ctx := context.Background()

    // 方式 1：本地文件
    fileLoader := feature.NewFileMetadataLoader()
    meta1, _ := fileLoader.Load(ctx, "model/feature_meta.json")

    // 方式 2：HTTP 接口
    httpLoader := feature.NewHTTPMetadataLoader(5 * time.Second)
    meta2, _ := httpLoader.Load(ctx, "http://api.example.com/models/v1.0.0/feature_meta")

    // 方式 3：S3 兼容协议（支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等）
    s3Client := &MyS3Client{} // 可以是 AWSS3Client、AliyunOSSClient、TencentCOSClient 等
    s3Loader := feature.NewS3MetadataLoader(s3Client, "my-bucket")
    meta3, _ := s3Loader.Load(ctx, "models/v1.0.0/feature_meta.json")
}
```

## 运行示例

```bash
cd examples/feature_metadata_loader
go run main.go
```
