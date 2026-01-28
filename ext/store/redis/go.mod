module github.com/rushteam/reckit/ext/store/redis

go 1.25.5

require (
	github.com/bits-and-blooms/bloom/v3 v3.7.1
	github.com/redis/go-redis/v9 v9.5.1
	github.com/rushteam/reckit v0.0.0
)

require (
	github.com/bits-and-blooms/bitset v1.24.2 // indirect
	github.com/cespare/xxhash/v2 v2.2.0 // indirect
	github.com/dgryski/go-rendezvous v0.0.0-20200823014737-9f7001d12a5f // indirect
	github.com/kr/text v0.2.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

replace github.com/rushteam/reckit => ../../../
