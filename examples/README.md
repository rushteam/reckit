# Examples 示例项目

本目录包含 Reckit 的各种使用示例，使用独立的 `go.mod` 文件管理依赖。

## 目录结构

```
examples/
├── go.mod              # 独立的 go.mod 文件
├── README.md          # 本文件
├── feedback/          # 反馈收集系统示例
│   ├── collector.go
│   ├── kafka_collector.go
│   ├── hook.go
│   ├── main.go
│   └── README.md
└── ...                # 其他示例
```

## 使用方式

### 1. 初始化依赖

在 `examples` 目录下运行：

```bash
cd examples
go mod tidy
```

这会自动安装所有依赖，包括：
- `github.com/rushteam/reckit`（主项目，通过 replace 指向本地）
- 示例项目需要的其他依赖（如 `franz-go`）

### 2. 运行示例

```bash
# 运行反馈收集示例
go run feedback/main.go

# 运行其他示例
go run basic/main.go
go run full_recommendation_system/main.go
```

## 优势

1. **依赖隔离**：示例项目的依赖不会影响主项目
2. **独立版本管理**：示例可以独立管理依赖版本
3. **易于测试**：可以独立运行和测试示例代码
4. **不侵入主项目**：主项目的 `go.mod` 保持简洁

## 注意事项

- 示例项目通过 `replace` 指令引用本地主项目
- 如果需要发布示例项目，需要移除 `replace` 指令或使用实际的主项目版本
- 每个示例目录都是独立的包，可以独立运行
