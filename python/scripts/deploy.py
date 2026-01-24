#!/usr/bin/env python3
"""
模型部署脚本

从对象存储拉取指定版本的模型，更新到推理服务，并触发 reload。
"""
import argparse
import os
import sys
import tarfile
import requests

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def deploy_model(version: str, model_dir: str, service_url: str = None, s3_bucket: str = None):
    """
    部署模型到推理服务
    
    Args:
        version: 模型版本
        model_dir: 模型目录
        service_url: 推理服务 URL（用于触发 reload）
        s3_bucket: S3 桶名（可选，若未配置则从本地加载）
    """
    print(f"部署模型: {version}")
    
    # 1. 从 S3 拉取模型（如果配置了）
    if s3_bucket:
        try:
            import boto3
            
            s3 = boto3.client("s3")
            s3_key = f"{version}/model.tar.gz"
            tar_path = os.path.join(model_dir, f"model-{version}.tar.gz")
            
            print(f"从 S3 拉取模型: s3://{s3_bucket}/{s3_key}")
            s3.download_file(s3_bucket, s3_key, tar_path)
            print(f"✅ 模型已下载: {tar_path}")
        except ImportError:
            print("⚠️  boto3 未安装，跳过 S3 下载")
            tar_path = None
        except Exception as e:
            print(f"⚠️  S3 下载失败: {e}")
            tar_path = None
    else:
        # 从本地查找
        tar_path = os.path.join(model_dir, f"model-{version}.tar.gz")
        if not os.path.exists(tar_path):
            print(f"⚠️  本地模型文件不存在: {tar_path}")
            tar_path = None
    
    # 2. 解压模型
    if tar_path and os.path.exists(tar_path):
        print(f"解压模型: {tar_path}")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=model_dir)
        print(f"✅ 模型已解压到: {model_dir}")
    else:
        print("⚠️  跳过模型文件更新（使用现有模型）")
    
    # 3. 更新当前版本
    current_version_path = os.path.join(model_dir, ".current_version")
    with open(current_version_path, "w") as f:
        f.write(version)
    print(f"✅ 当前版本已更新: {version}")
    
    # 4. 触发服务 reload（如果配置了）
    if service_url:
        reload_url = f"{service_url}/reload"
        print(f"触发服务 reload: {reload_url}")
        
        try:
            response = requests.post(reload_url, timeout=30)
            response.raise_for_status()
            result = response.json()
            print(f"✅ 服务 reload 成功:")
            print(f"   旧版本: {result.get('old_version', 'unknown')}")
            print(f"   新版本: {result.get('new_version', 'unknown')}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  服务 reload 失败: {e}")
            print("   请手动重启服务或检查服务状态")
    else:
        print("⚠️  未配置服务 URL，请手动触发 reload 或重启服务")
        print(f"   手动触发: curl -X POST {service_url or 'http://localhost:8080'}/reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型部署")
    parser.add_argument("--version", required=True, help="模型版本")
    parser.add_argument("--model-dir", default="/app/model", help="模型目录")
    parser.add_argument("--service-url", default=None, help="推理服务 URL（用于触发 reload）")
    parser.add_argument("--s3-bucket", default=None, help="S3 桶名（可选）")
    
    args = parser.parse_args()
    
    try:
        deploy_model(args.version, args.model_dir, args.service_url, args.s3_bucket)
    except Exception as e:
        print(f"❌ 模型部署失败: {e}", file=sys.stderr)
        sys.exit(1)
