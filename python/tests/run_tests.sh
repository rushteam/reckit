#!/bin/bash
# 运行测试脚本

echo "运行 Python 测试..."

# 添加项目根目录到路径
export PYTHONPATH="${PYTHONPATH}:$(pwd)/.."

# 运行单元测试
python -m unittest discover -s tests -p "test_*.py" -v

echo "测试完成！"
