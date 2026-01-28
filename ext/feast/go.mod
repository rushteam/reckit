module github.com/rushteam/reckit/ext/feast

go 1.25.5

require (
	github.com/feast-dev/feast/sdk/go v0.9.4
	github.com/rushteam/reckit v0.0.0
)

require (
	cloud.google.com/go v0.62.0 // indirect
	github.com/golang/groupcache v0.0.0-20200121045136-8c9f03a8e57e // indirect
	github.com/golang/protobuf v1.5.0 // indirect
	github.com/opentracing-contrib/go-grpc v0.0.0-20200813121455-4a6760c71486 // indirect
	github.com/opentracing/opentracing-go v1.1.0 // indirect
	go.opencensus.io v0.22.4 // indirect
	golang.org/x/net v0.0.0-20200707034311-ab3426394381 // indirect
	golang.org/x/oauth2 v0.0.0-20200107190931-bf48bf16ab8d // indirect
	golang.org/x/sys v0.0.0-20200803210538-64077c9b5642 // indirect
	golang.org/x/text v0.3.3 // indirect
	google.golang.org/api v0.30.0 // indirect
	google.golang.org/appengine v1.6.6 // indirect
	google.golang.org/genproto v0.0.0-20200804131852-c06518451d9c // indirect
	google.golang.org/grpc v1.32.0 // indirect
	google.golang.org/protobuf v1.34.2 // indirect
)

replace github.com/rushteam/reckit => ../../

// 解决 genproto 歧义：强制使用旧版单体 genproto，排除新版独立子模块
exclude google.golang.org/genproto/googleapis/rpc v0.0.0-20240826202546-f6391c0de4c7
