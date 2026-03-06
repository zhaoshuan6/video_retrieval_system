#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用MOT17数据集测试视频预处理模块
"""

import os
import sys
from pathlib import Path

# === 设置环境变量 ===
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# === 添加项目根目录到Python路径 ===
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# === 现在导入video_processor ===
# 方法：直接用文件路径导入
import importlib.util

video_processor_path = project_root / "backend" / "preprocessing" / "video_processor.py"
spec = importlib.util.spec_from_file_location("video_processor", video_processor_path)
video_processor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(video_processor_module)

# 获取VideoProcessor类
VideoProcessor = video_processor_module.VideoProcessor

print("✅ VideoProcessor导入成功！")


def convert_mot17_to_video():
    """
    将MOT17的图片序列转换为视频
    """
    import cv2
    from tqdm import tqdm

    mot17_dir = Path("data/MOT17/train")

    # 查找可用的序列
    if not mot17_dir.exists():
        print("❌ MOT17数据集不存在")
        print(f"预期路径: {mot17_dir.absolute()}")
        print("\n请先下载MOT17数据集")
        return None

    sequences = sorted([d for d in mot17_dir.iterdir() if d.is_dir()])

    if not sequences:
        print("❌ 未找到MOT17序列")
        return None

    # 使用第一个序列
    seq = sequences[0]
    print(f"\n使用序列: {seq.name}")

    img_dir = seq / "img1"
    if not img_dir.exists():
        print(f"❌ 图片目录不存在: {img_dir}")
        return None

    # 获取所有图片
    images = sorted(img_dir.glob("*.jpg"))

    if not images:
        print("❌ 未找到图片")
        return None

    print(f"图片数量: {len(images)}")

    # 创建视频
    output_dir = Path("data/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / f"{seq.name}.mp4"

    if output_video.exists():
        print(f"✅ 视频已存在: {output_video}")
        return output_video

    print(f"\n📹 创建视频: {output_video.name}")

    # 读取第一张图片获取尺寸
    first_img = cv2.imread(str(images[0]))
    height, width = first_img.shape[:2]

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  # MOT17默认30fps
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    # 写入所有帧
    for img_path in tqdm(images, desc="创建视频"):
        frame = cv2.imread(str(img_path))
        out.write(frame)

    out.release()

    print(f"✅ 视频创建完成: {output_video}")
    return output_video


def main():
    """主测试函数"""
    print("="*70)
    print("  MOT17数据集预处理测试")
    print("="*70)

    # 检查MOT17数据集
    mot17_dir = Path("data/MOT17/train")
    if not mot17_dir.exists():
        print("\n❌ MOT17数据集不存在")
        print(f"预期路径: {mot17_dir.absolute()}")
        print("\n请先下载MOT17数据集")
        print("可以手动下载：https://motchallenge.net/data/MOT17/")
        print(f"然后解压到: {mot17_dir.parent.absolute()}")
        return

    # 将MOT17图片序列转换为视频
    video_path = convert_mot17_to_video()

    if video_path is None:
        return

    # 创建视频处理器
    print("\n" + "="*70)
    print("  初始化视频处理器")
    print("="*70)

    processor = VideoProcessor(device="cuda")

    # 处理视频
    print("\n" + "="*70)
    print("  开始处理视频")
    print("="*70)

    result = processor.process_video(
        video_path=video_path,
        output_base_dir="data/processed",
        interval=10  # 每10秒提取一帧
    )

    print("\n" + "="*70)
    print("  处理完成")
    print("="*70)
    print(f"\n视频名称: {result['video_name']}")
    print(f"关键帧数: {result['keyframes']}")
    print(f"检测人物: {result['total_persons']}")
    print(f"输出文件: {result['output_file']}")

    # 查看结果
    print("\n" + "="*70)
    print("  查看处理结果")
    print("="*70)

    import pickle

    with open(result['output_file'], 'rb') as f:
        processed_data = pickle.load(f)

    print(f"\n处理了 {len(processed_data)} 帧")

    for i, frame_data in enumerate(processed_data[:3]):  # 只显示前3帧
        print(f"\n帧 {i+1}:")
        print(f"  图片: {Path(frame_data['frame_path']).name}")
        print(f"  检测到的人物: {len(frame_data['persons'])}")

        for j, person in enumerate(frame_data['persons'][:3]):  # 每帧最多显示3个人
            print(f"    人物 {j+1}:")
            print(f"      位置: ({person['bbox'][0]:.0f}, {person['bbox'][1]:.0f}) - ({person['bbox'][2]:.0f}, {person['bbox'][3]:.0f})")
            print(f"      置信度: {person['confidence']:.2%}")
            print(f"      特征维度: {person['features'].shape}")

    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
