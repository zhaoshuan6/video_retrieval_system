#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载MOT17数据集
"""

import os
import requests
from pathlib import Path
from tqdm import tqdm
import zipfile

def download_file(url, output_path):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    return output_path

def extract_zip(zip_path, extract_to):
    """解压ZIP文件"""
    print(f"\n📦 正在解压: {zip_path.name}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in tqdm(zip_ref.namelist(), desc="解压进度"):
            zip_ref.extract(file, extract_to)
    print("✅ 解压完成！")

def main():
    """主函数"""
    print("="*60)
    print("  MOT17数据集下载")
    print("="*60)

    # 数据集URL（从MOTChallenge官网）
    # 注意：MOT17数据集大约1.5GB
    url = "https://motchallenge.net/data/MOT17.zip"

    # 备用URL（如果官网下载失败）
    # url = "https://drive.google.com/uc?id=..." # Google Drive链接

    output_dir = Path("data")
    zip_path = output_dir / "MOT17.zip"

    # 下载
    if not zip_path.exists():
        print(f"\n📥 开始下载MOT17数据集...")
        print(f"   URL: {url}")
        print(f"   大小: 约1.5GB")
        print(f"   目标: {zip_path}")

        try:
            download_file(url, zip_path)
            print("✅ 下载完成！")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            print("\n请手动下载:")
            print("1. 访问: https://motchallenge.net/data/MOT17/")
            print("2. 下载 MOT17.zip")
            print(f"3. 保存到: {zip_path.absolute()}")
            return
    else:
        print(f"✅ 文件已存在: {zip_path}")

    # 解压
    extract_to = output_dir / "MOT17"
    if not extract_to.exists():
        extract_zip(zip_path, output_dir)
    else:
        print(f"✅ 已解压到: {extract_to}")

    # 查看数据集结构
    print("\n" + "="*60)
    print("  数据集结构")
    print("="*60)

    train_dir = extract_to / "train"
    if train_dir.exists():
        sequences = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        print(f"\n训练序列数量: {len(sequences)}")
        print("\n序列列表:")
        for seq in sequences:
            seq_path = train_dir / seq
            img_dir = seq_path / "img1"
            if img_dir.exists():
                num_frames = len(list(img_dir.glob("*.jpg")))
                print(f"  - {seq}: {num_frames} 帧")

    print("\n" + "="*60)
    print("✅ MOT17数据集准备完成！")
    print("="*60)

if __name__ == "__main__":
    main()
