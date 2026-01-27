module github.com/rushteam/reckit/ext/vector/milvus

go 1.25.5

require (
	github.com/milvus-io/milvus-sdk-go/v2 v2.3.4
	github.com/rushteam/reckit v0.0.0
)

require github.com/milvus-io/milvus/client/v2 v2.6.2 // indirect

replace github.com/rushteam/reckit => ../../../
