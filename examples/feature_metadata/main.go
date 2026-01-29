package main

import (
	"context"
	"fmt"
	"log"
	"path/filepath"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/feature"
	"github.com/rushteam/reckit/model"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/rank"
	"github.com/rushteam/reckit/recall"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// ========== 方式 1：使用本地文件加载器（推荐用于开发环境）==========
	fmt.Println("=== 方式 1：本地文件加载 ===")
	projectRoot := "../../"
	metaPath := filepath.Join(projectRoot, "python/model/feature_meta.json")
	scalerPath := filepath.Join(projectRoot, "python/model/feature_scaler.json")

	fileMetaLoader := feature.NewFileMetadataLoader()
	fileScalerLoader := feature.NewFileScalerLoader()

	meta, err := fileMetaLoader.Load(ctx, metaPath)
	if err != nil {
		log.Printf("加载特征元数据失败（可选）: %v", err)
		log.Println("继续运行，但无法进行特征验证...")
	} else {
		fmt.Printf("✅ 特征元数据加载成功\n")
		fmt.Printf("  模型版本: %s\n", meta.ModelVersion)
		fmt.Printf("  特征数量: %d\n", meta.FeatureCount)
		fmt.Printf("  是否标准化: %v\n", meta.Normalized)
		fmt.Printf("  特征列: %v\n", meta.FeatureColumns)
		fmt.Println()
	}

	var scaler feature.FeatureScaler
	if meta != nil && meta.Normalized {
		scaler, err = fileScalerLoader.Load(ctx, scalerPath)
		if err != nil {
			log.Printf("⚠️  特征标准化器加载失败（模型需要标准化）: %v", err)
		} else {
			fmt.Printf("✅ 特征标准化器加载成功\n")
			fmt.Printf("  标准化特征数: %d\n", len(scaler))
			fmt.Println()
		}
	}

	// ========== 方式 2：使用 HTTP 接口加载器（推荐用于生产环境）==========
	fmt.Println("=== 方式 2：HTTP 接口加载（示例） ===")
	
	// 示例：从 HTTP 接口加载（需要实际的服务地址）
	// 注意：这里使用示例 URL，实际使用时需要替换为真实的服务地址
	//
	// httpMetaLoader := feature.NewHTTPMetadataLoader(5 * time.Second)
	// httpScalerLoader := feature.NewHTTPScalerLoader(5 * time.Second)
	//
	// metaHTTP, err := httpMetaLoader.Load(ctx, "http://api.example.com/models/v1.0.0/feature_meta")
	// if err != nil {
	//     log.Printf("从 HTTP 加载失败: %v", err)
	// } else {
	//     fmt.Printf("✅ 从 HTTP 加载成功: %s\n", metaHTTP.ModelVersion)
	// }

	fmt.Println("（HTTP 加载示例已注释，需要实际的服务地址）")
	fmt.Println("（取消注释上面的代码并替换 URL 即可使用）")
	fmt.Println()

	// ========== 方式 3：使用 S3 兼容协议加载器（推荐用于云环境）==========
	fmt.Println("=== 方式 3：S3 兼容协议加载（示例） ===")
	// S3 兼容协议支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等
	// 需要实现 feature.S3Client 接口
	// type MyS3Client struct {
	//     // 实现 S3Client 接口
	// }
	// s3Client := &MyS3Client{...}
	// s3MetaLoader := feature.NewS3MetadataLoader(s3Client, "my-bucket")
	// s3ScalerLoader := feature.NewS3ScalerLoader(s3Client, "my-bucket")
	//
	// metaS3, err := s3MetaLoader.Load(ctx, "models/v1.0.0/feature_meta.json")
	// if err != nil {
	//     log.Printf("从 S3 兼容存储加载失败: %v", err)
	// } else {
	//     fmt.Printf("✅ 从 S3 兼容存储加载成功: %s\n", metaS3.ModelVersion)
	// }

	fmt.Println("（S3 兼容协议加载示例已注释，需要实现 S3Client 接口）")
	fmt.Println("（支持 AWS S3、阿里云 OSS、腾讯云 COS、MinIO 等）")
	fmt.Println()

	// ========== 使用加载的特征元数据进行 Pipeline ==========
	fmt.Println("=== 运行 Pipeline ===")

	// 创建特征注入节点
	enrichNode := &feature.EnrichNode{
		UserFeaturePrefix:  "user_",
		ItemFeaturePrefix:  "item_",
		CrossFeaturePrefix: "cross_",
		KeyUserFeatures:    []string{"age", "gender"},
		KeyItemFeatures:    []string{"ctr", "cvr", "price"},
	}

	// 创建 RPC 模型
	xgbModel := model.NewRPCModel("xgboost", "http://localhost:8080/predictions/xgb", 5*time.Second)

	// 构建 Pipeline
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					&recall.Hot{IDs: []string{"1", "2", "3"}},
				},
			},
			enrichNode,
			// 可选：在 RPCNode 之前添加特征验证节点
			&featureValidationNode{
				meta:   meta,
				scaler: scaler,
			},
			&rank.RPCNode{Model: xgbModel},
		},
	}

	// 创建用户上下文
	rctx := &core.RecommendContext{
		UserID: "user_123",
		Scene:  "feed",
		UserProfile: map[string]any{
			"age":    25.0,
			"gender": 1.0,
		},
	}

	// 运行 Pipeline
	items, err := p.Run(ctx, rctx, nil)
	if err != nil {
		log.Fatalf("Pipeline 运行失败: %v", err)
	}

	// 显示结果
	fmt.Println("推荐结果:")
	for i, it := range items {
		fmt.Printf("#%d id=%s score=%.4f\n", i+1, it.ID, it.Score)
		if meta != nil {
			// 验证特征完整性
			missing := meta.GetMissingFeatures(it.Features)
			if len(missing) > 0 {
				fmt.Printf("  ⚠️  缺失特征: %v\n", missing)
			}
		}
	}
}

// featureValidationNode 特征验证节点（可选，用于在发送到 Python 服务前验证特征）
type featureValidationNode struct {
	meta   *feature.FeatureMetadata
	scaler feature.FeatureScaler
}

func (n *featureValidationNode) Name() string { return "feature.validation" }
func (n *featureValidationNode) Kind() pipeline.Kind { return pipeline.KindPostProcess }

func (n *featureValidationNode) Process(
	ctx context.Context,
	rctx *core.RecommendContext,
	items []*core.Item,
) ([]*core.Item, error) {
	if n.meta == nil {
		// 如果没有加载元数据，跳过验证
		return items, nil
	}

	for _, item := range items {
		if item == nil {
			continue
		}

		// 验证特征完整性
		missing := n.meta.GetMissingFeatures(item.Features)
		if len(missing) > 0 {
			// 填充缺失特征为 0.0
			for _, col := range missing {
				item.Features[col] = 0.0
			}
		}

		// 可选：在 Go 端进行标准化（如果需要在 Go 端做标准化）
		// 注意：当前架构标准化在 Python 服务中完成，这里仅作示例
		// if n.scaler != nil && n.meta.Normalized {
		//     item.Features = n.scaler.Normalize(item.Features)
		// }
	}

	return items, nil
}
