#!/usr/bin/env python3
"""
模型注册脚本

将训练好的模型打包上传到对象存储（S3/OSS），并在本地记录版本信息。
"""
import argparse
import json
import os
import sys
import tarfile
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


def register_model(version: str, model_dir: str, s3_bucket: str = None):
    """
    注册模型到对象存储和本地注册中心
    
    Args:
        version: 模型版本
        model_dir: 模型目录
        s3_bucket: S3 桶名（可选，若未配置则只本地注册）
    """
    print(f"注册模型: {version}")
    
    # 检查模型文件
    model_files = [
        "xgb_model.json",
        "feature_meta.json",
        "deepfm_model.pt",
        "deepfm_feature_meta.json",
    ]
    
    existing_files = []
    for f in model_files:
        path = os.path.join(model_dir, f)
        if os.path.exists(path):
            existing_files.append(f)
    
    if not existing_files:
        raise FileNotFoundError(f"模型目录中没有找到任何模型文件: {model_dir}")
    
    print(f"找到模型文件: {existing_files}")
    
    # 打包模型
    tar_path = os.path.join(model_dir, f"model-{version}.tar.gz")
    print(f"打包模型: {tar_path}")
    
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in existing_files:
            file_path = os.path.join(model_dir, f)
            tar.add(file_path, arcname=f)
            print(f"  添加: {f}")
    
    print(f"✅ 模型打包完成: {tar_path}")
    
    # 上传到 S3/OSS（如果配置了）
    if s3_bucket:
        try:
            import boto3
            
            s3 = boto3.client("s3")
            s3_key = f"{version}/model.tar.gz"
            
            print(f"上传到 S3: s3://{s3_bucket}/{s3_key}")
            s3.upload_file(tar_path, s3_bucket, s3_key)
            print(f"✅ 模型已上传到 S3: s3://{s3_bucket}/{s3_key}")
        except ImportError:
            print("⚠️  boto3 未安装，跳过 S3 上传")
        except Exception as e:
            print(f"⚠️  S3 上传失败: {e}")
    
    # 写入本地注册中心
    registry_path = os.path.join(model_dir, "registry.json")
    registry = {}
    
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            registry = json.load(f)
    
    registry[version] = {
        "version": version,
        "files": existing_files,
        "registered_at": datetime.now().isoformat(),
        "status": "staging",  # staging / production / archived
        "s3_bucket": s3_bucket,
        "s3_key": f"{version}/model.tar.gz" if s3_bucket else None,
    }
    
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)
    
    print(f"✅ 模型已注册到本地注册中心: {registry_path}")
    print(f"   版本: {version}")
    print(f"   状态: staging")
    print(f"   文件数: {len(existing_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型注册")
    parser.add_argument("--version", required=True, help="模型版本")
    parser.add_argument("--model-dir", default="/app/model", help="模型目录")
    parser.add_argument("--s3-bucket", default=None, help="S3 桶名（可选）")
    
    args = parser.parse_args()
    
    try:
        register_model(args.version, args.model_dir, args.s3_bucket)
    except Exception as e:
        print(f"❌ 模型注册失败: {e}", file=sys.stderr)
        sys.exit(1)
