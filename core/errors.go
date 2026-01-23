package core

// DomainError 是领域层的统一错误类型。
//
// 设计原则：
//   - 所有领域层错误都使用此类型
//   - 提供错误代码（Code）和消息（Message）
//   - 支持错误检查函数（IsXXX）
//
// 使用场景：
//   - Store 错误：NOT_FOUND, NOT_SUPPORTED
//   - Feature 错误：FEATURE_NOT_FOUND, SERVICE_UNAVAILABLE
//   - Vector 错误：NOT_SUPPORTED
//   - 其他领域错误
type DomainError struct {
	Code    string // 错误代码（如 "NOT_FOUND", "NOT_SUPPORTED"）
	Message string // 错误消息
	Module  string // 模块名称（如 "store", "feature", "vector"）
}

func (e *DomainError) Error() string {
	return e.Message
}

// IsDomainError 检查错误是否为 DomainError 类型
func IsDomainError(err error) bool {
	if err == nil {
		return false
	}
	_, ok := err.(*DomainError)
	return ok
}

// GetDomainError 获取 DomainError，如果不是则返回 nil
func GetDomainError(err error) *DomainError {
	if err == nil {
		return nil
	}
	if domainErr, ok := err.(*DomainError); ok {
		return domainErr
	}
	return nil
}

// NewDomainError 创建新的领域错误
func NewDomainError(module, code, message string) *DomainError {
	return &DomainError{
		Module:  module,
		Code:    code,
		Message: message,
	}
}

// 错误代码常量
const (
	// 通用错误代码
	ErrorCodeNotFound      = "NOT_FOUND"      // 资源不存在
	ErrorCodeNotSupported  = "NOT_SUPPORTED"  // 操作不支持
	ErrorCodeUnavailable   = "UNAVAILABLE"    // 服务不可用
	ErrorCodeInvalidInput  = "INVALID_INPUT"  // 输入无效
	ErrorCodeInternalError = "INTERNAL_ERROR" // 内部错误
)

// 模块名称常量
const (
	ModuleStore   = "store"   // 存储模块
	ModuleFeature = "feature" // 特征模块
	ModuleVector  = "vector"  // 向量模块
	ModuleService = "service" // 服务模块
)

// 通用错误检查函数

// IsNotFound 检查错误是否为 NOT_FOUND
func IsNotFound(err error) bool {
	if err == nil {
		return false
	}
	if domainErr := GetDomainError(err); domainErr != nil {
		return domainErr.Code == ErrorCodeNotFound
	}
	return false
}

// IsNotSupported 检查错误是否为 NOT_SUPPORTED
func IsNotSupported(err error) bool {
	if err == nil {
		return false
	}
	if domainErr := GetDomainError(err); domainErr != nil {
		return domainErr.Code == ErrorCodeNotSupported
	}
	return false
}

// IsUnavailable 检查错误是否为 UNAVAILABLE
func IsUnavailable(err error) bool {
	if err == nil {
		return false
	}
	if domainErr := GetDomainError(err); domainErr != nil {
		return domainErr.Code == ErrorCodeUnavailable
	}
	return false
}
