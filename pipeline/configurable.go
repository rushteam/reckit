package pipeline

import (
	"context"

	"github.com/rushteam/reckit/core"
)

// Configurable 是 Node 的可选接口，用于支持运行时动态配置注入。
//
// 当 Node 实现此接口时，Pipeline runner 在调用 Process 前会自动调用 ApplyConfig，
// 使 Node 有机会从 RecommendContext 中读取场景级覆盖参数并调整自身行为。
//
// 典型使用场景：
//   - 多样性 Node 根据场景配置动态调整 DiversityKeys / MaxConsecutive
//   - TopN Node 根据场景配置调整截断数量
//   - 召回 Node 根据场景配置选择不同的分发策略
//
// 示例：
//
//	type MyNode struct { TopK int }
//
//	func (n *MyNode) ApplyConfig(ctx context.Context, rctx *core.RecommendContext) error {
//	    if cfg, ok := core.ExtensionAs[*SceneConfig](rctx, "scene_config"); ok {
//	        if override := cfg.GetTopK(n.Name()); override > 0 {
//	            n.TopK = override
//	        }
//	    }
//	    return nil
//	}
type Configurable interface {
	// ApplyConfig 在 Process 前被 Pipeline runner 调用。
	// Node 可从 rctx 中读取场景配置并调整自身参数。
	// 返回 error 时按 ErrorHook 策略处理（同 Process 错误）。
	ApplyConfig(ctx context.Context, rctx *core.RecommendContext) error
}
