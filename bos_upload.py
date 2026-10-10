"""
上传文件/目录到百度 BOS。

用法:
    python bos_upload.py <本地路径> <云端key或前缀> --ak <AccessKey> --sk <SecretKey> [--bucket 桶名] [-r] [--list]

示例:
    # 上传单个文件
    python bos_upload.py ./data.h5 S2/TeleopData/YAM/TASK-YAM-0002/data.h5 --ak <AK> --sk <SK>

    # 递归上传整个目录
    python bos_upload.py ./TASK-YAM-0002/ S2/TeleopData/YAM/TASK-YAM-0002/ --ak <AK> --sk <SK> -r

    # 上传后列一下目标前缀验证
    python bos_upload.py ./data.h5 S2/TeleopData/YAM/TASK-YAM-0002/data.h5 --ak <AK> --sk <SK> --list
"""

from __future__ import annotations

import argparse
import os

from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.services.bos.bos_client import BosClient

# 默认值(都不是密钥,可改;密钥通过命令行 --ak/--sk 传入)
ENDPOINT = "bj.bcebos.com"
BUCKET = "rhos-baige-general-beijing-prod"   # 注意:你参考代码是 general 桶,
                                             # 之前问下载的是 confidential 桶,按目标地址改


def make_client(ak: str, sk: str, endpoint: str) -> BosClient:
    config = BceClientConfiguration(
        credentials=BceCredentials(ak, sk), endpoint=endpoint
    )
    return BosClient(config)


def upload_file(client: BosClient, local_path: str, key: str, bucket: str) -> None:
    """上传单个文件。key 是云端完整路径(含文件名),不带 bos:// 和 bucket 名。"""
    client.put_object_from_file(bucket, key, local_path)
    print(f"✅ {local_path} -> bos://{bucket}/{key}")


def upload_dir(client: BosClient, local_dir: str, prefix: str, bucket: str) -> int:
    """递归上传整个目录。prefix 是云端前缀(目录),结尾有没有 / 都行。"""
    local_dir = os.path.abspath(local_dir)
    prefix = prefix.strip("/")
    count = 0
    for root, _dirs, files in os.walk(local_dir):
        for name in files:
            local_path = os.path.join(root, name)
            rel = os.path.relpath(local_path, local_dir).replace("\\", "/")
            key = f"{prefix}/{rel}" if prefix else rel
            client.put_object_from_file(bucket, key, local_path)
            count += 1
            print(f"  [{count}] {key}")
    print(f"✅ 目录上传完成,共 {count} 个文件 -> bos://{bucket}/{prefix}/")
    return count


def list_prefix(client: BosClient, prefix: str, bucket: str) -> None:
    """列出目标前缀下的文件,用于上传后验证。"""
    response = client.list_objects(bucket, prefix=prefix)
    print(f"\n目标 [{prefix}] 下现有文件:")
    for content in response.contents:
        print(f"  {content.key}  ({content.size} Bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload files to Baidu BOS")
    parser.add_argument("local_path", help="本地文件或目录")
    parser.add_argument("key_or_prefix", help="云端 key(文件)或前缀(目录)")
    parser.add_argument("--ak", required=True, help="百度云 AccessKey ID")
    parser.add_argument("--sk", required=True, help="百度云 SecretKey")
    parser.add_argument("--endpoint", default=ENDPOINT, help=f"BOS endpoint(默认 {ENDPOINT})")
    parser.add_argument("--bucket", default=BUCKET, help=f"桶名(默认 {BUCKET})")
    parser.add_argument("-r", "--recursive", action="store_true", help="上传目录")
    parser.add_argument("--list", action="store_true", help="上传后列出目标前缀验证")
    args = parser.parse_args()

    client = make_client(args.ak, args.sk, args.endpoint)

    if args.recursive or os.path.isdir(args.local_path):
        upload_dir(client, args.local_path, args.key_or_prefix, args.bucket)
    else:
        upload_file(client, args.local_path, args.key_or_prefix, args.bucket)

    if args.list:
        list_prefix(client, args.key_or_prefix.rstrip("/"), args.bucket)


if __name__ == "__main__":
    main()
