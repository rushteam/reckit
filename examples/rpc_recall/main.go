package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/rushteam/reckit/core"
	"github.com/rushteam/reckit/pipeline"
	"github.com/rushteam/reckit/recall"
)

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// 示例 1：使用默认请求/响应格式的 RPC 召回
	fmt.Println("=== 示例 1：默认格式的 RPC 召回 ===")
	rpcRecall1 := recall.NewRPCRecall(
		"http://localhost:8080/recall", // 召回服务端点
		2*time.Second,                  // 超时时间
	).WithTopK(20) // 设置 TopK

	// 在 Fanout 中使用
	fanout := &recall.Fanout{
		Sources: []recall.Source{
			rpcRecall1,
			&recall.Hot{IDs: []string{"1", "2", "3", "4", "5"}}, // 本地热门召回作为备用
		},
		Dedup:         true,
		Timeout:       2 * time.Second,
		MaxConcurrent: 5,
		MergeStrategy: &recall.PriorityMergeStrategy{},
	}

	// 创建推荐上下文
	rctx := &core.RecommendContext{
		UserID: "user_123",
		Scene:  "feed",
		UserProfile: map[string]any{
			"age":    25.0,
			"gender": 1.0,
			"city":   "beijing",
		},
		Realtime: map[string]any{
			"hour":   float64(time.Now().Hour()),
			"device": "mobile",
		},
	}

	// 执行召回
	items, err := fanout.Process(ctx, rctx, nil)
	if err != nil {
		log.Printf("召回失败: %v", err)
	} else {
		fmt.Printf("召回结果数量: %d\n", len(items))
		for i, item := range items {
			if i >= 5 {
				break
			}
			fmt.Printf("  Item %d: ID=%s, Score=%.3f, Source=%s\n",
				i+1, item.ID, item.Score,
				item.Labels["recall_source"].Value)
		}
	}

	fmt.Println()

	// 示例 2：使用自定义请求/响应格式的 RPC 召回
	fmt.Println("=== 示例 2：自定义格式的 RPC 召回 ===")
	rpcRecall2 := recall.NewRPCRecall(
		"http://localhost:8080/api/v1/recommend",
		3*time.Second,
	).
		WithTopK(30).
		// 自定义请求构建器
		WithRequestBuilder(func(ctx context.Context, rctx *core.RecommendContext, topK int) (map[string]interface{}, error) {
			// 自定义请求格式
			return map[string]interface{}{
				"userId": rctx.UserID,
				"limit":  topK,
				"scene":  rctx.Scene,
				"extra": map[string]interface{}{
					"userProfile": rctx.UserProfile,
					"realtime":    rctx.Realtime,
				},
			}, nil
		}).
		// 自定义响应解析器
		WithResponseParser(func(resp *http.Response) ([]*core.Item, error) {
			// 自定义响应格式解析
			var result struct {
				Data []struct {
					ItemID string  `json:"itemId"`
					Score  float64 `json:"score"`
				} `json:"data"`
			}

			if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
				return nil, err
			}

			items := make([]*core.Item, 0, len(result.Data))
			for _, item := range result.Data {
				it := core.NewItem(item.ItemID)
				it.Score = item.Score
				items = append(items, it)
			}
			return items, nil
		})

	// 在 Pipeline 中使用
	p := &pipeline.Pipeline{
		Nodes: []pipeline.Node{
			&recall.Fanout{
				Sources: []recall.Source{
					rpcRecall2,
					&recall.Hot{IDs: []string{"1", "2", "3"}},
				},
				Dedup:         true,
				Timeout:       3 * time.Second,
				MaxConcurrent: 3,
				MergeStrategy: &recall.PriorityMergeStrategy{},
			},
		},
	}

	items2, err := p.Run(ctx, rctx, nil)
	if err != nil {
		log.Printf("Pipeline 执行失败: %v", err)
	} else {
		fmt.Printf("Pipeline 召回结果数量: %d\n", len(items2))
	}

	fmt.Println()

	// 示例 3：直接使用 RPC 召回（不通过 Fanout）
	fmt.Println("=== 示例 3：直接使用 RPC 召回 ===")
	rpcRecall3 := recall.NewRPCRecall(
		"http://localhost:8080/recall",
		2*time.Second,
	).WithTopK(10)

	items3, err := rpcRecall3.Recall(ctx, rctx)
	if err != nil {
		log.Printf("RPC 召回失败: %v", err)
	} else {
		fmt.Printf("直接召回结果数量: %d\n", len(items3))
		for i, item := range items3 {
			if i >= 3 {
				break
			}
			fmt.Printf("  Item %d: ID=%s, Score=%.3f\n", i+1, item.ID, item.Score)
		}
	}
}
