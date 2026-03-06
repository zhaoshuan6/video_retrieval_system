#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频检索系统 - 环境验证脚本
"""

import sys
from typing import Tuple

def print_header(title: str):
    """打印标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_test(name: str, step: int, total: int):
    """打印测试项"""
    print(f"\n[{step}/{total}] {name}")
    print("-" * 70)

def test_pytorch() -> Tuple[bool, str]:
    """测试PyTorch和CUDA"""
    try:
        import torch
        
        info = []
        info.append(f"PyTorch版本: {torch.__version__}")
        info.append(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            info.append(f"CUDA版本: {torch.version.cuda}")
            info.append(f"cuDNN版本: {torch.backends.cudnn.version()}")
            info.append(f"GPU设备: {torch.cuda.get_device_name(0)}")
            info.append(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            
            # 测试GPU计算
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            info.append("GPU计算测试: 通过")
            
            return True, "\n".join(info)
        else:
            return False, "\n".join(info) + "\n⚠️  CUDA不可用"
            
    except Exception as e:
        return False, f"错误: {str(e)}"

def test_clip() -> Tuple[bool, str]:
    """测试CLIP模型"""
    try:
        import torch
        import clip
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 加载模型（第一次会下载）
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        # 测试文本编码
        text = clip.tokenize(["a photo of a cat"]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        
        info = []
        info.append(f"CLIP模型: ViT-B/32")
        info.append(f"运行设备: {device}")
        info.append(f"文本编码输出: {text_features.shape}")
        info.append(f"特征维度: {text_features.shape[1]} (应为512)")
        
        return True, "\n".join(info)
        
    except Exception as e:
        return False, f"错误: {str(e)}"

def test_yolov8() -> Tuple[bool, str]:
    """测试YOLOv8"""
    try:
        from ultralytics import YOLO
        import torch
        
        # 只导入，不加载模型
        info = []
        info.append("Ultralytics YOLOv8")
        info.append(f"CUDA支持: {torch.cuda.is_available()}")
        
        return True, "\n".join(info)
        
    except Exception as e:
        return False, f"错误: {str(e)}"

def test_faiss() -> Tuple[bool, str]:
    """测试Faiss向量库"""
    try:
        import faiss
        import numpy as np
        import time
        
        info = []
        gpu_count = faiss.get_num_gpus()
        info.append(f"GPU数量: {gpu_count}")
        
        # 简单性能测试
        d = 512  # 维度
        nb = 10000  # 数据库大小
        nq = 100  # 查询数量
        
        np.random.seed(1234)
        xb = np.random.random((nb, d)).astype('float32')
        xq = np.random.random((nq, d)).astype('float32')
        
        # CPU索引
        index_cpu = faiss.IndexFlatL2(d)
        index_cpu.add(xb)
        
        start = time.time()
        D, I = index_cpu.search(xq, k=5)
        cpu_time = time.time() - start
        
        info.append(f"CPU检索耗时: {cpu_time*1000:.2f}ms")
        
        # GPU索引（如果可用）
        if gpu_count > 0:
            res = faiss.StandardGpuResources()
            index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            
            start = time.time()
            D, I = index_gpu.search(xq, k=5)
            gpu_time = time.time() - start
            
            info.append(f"GPU检索耗时: {gpu_time*1000:.2f}ms")
            info.append(f"GPU加速比: {cpu_time/gpu_time:.1f}x")
            
            return True, "\n".join(info)
        else:
            info.append("⚠️  GPU版本不可用，仅CPU")
            return True, "\n".join(info)
            
    except Exception as e:
        return False, f"错误: {str(e)}"

def test_ffmpeg() -> Tuple[bool, str]:
    """测试FFmpeg"""
    try:
        import imageio_ffmpeg
        import ffmpeg
        
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        
        info = []
        info.append(f"ffmpeg-python: 已安装")
        info.append(f"imageio-ffmpeg: 已安装")
        info.append(f"FFmpeg路径: {ffmpeg_path}")
        
        return True, "\n".join(info)
        
    except Exception as e:
        return False, f"错误: {str(e)}"

def test_flask() -> Tuple[bool, str]:
    """测试Flask"""
    try:
        import flask
        from flask_cors import CORS
        
        info = []
        info.append(f"Flask版本: {flask.__version__}")
        info.append(f"Flask-CORS: 已安装")
        
        return True, "\n".join(info)
        
    except Exception as e:
        return False, f"错误: {str(e)}"

def test_database() -> Tuple[bool, str]:
    """测试数据库驱动"""
    try:
        import pymysql
        import sqlalchemy
        
        info = []
        info.append(f"PyMySQL版本: {pymysql.__version__}")
        info.append(f"SQLAlchemy版本: {sqlalchemy.__version__}")
        
        return True, "\n".join(info)
        
    except Exception as e:
        return False, f"错误: {str(e)}"

def test_opencv() -> Tuple[bool, str]:
    """测试OpenCV和图像处理"""
    try:
        import cv2
        from PIL import Image
        import numpy as np
        
        info = []
        info.append(f"OpenCV版本: {cv2.__version__}")
        info.append(f"Pillow (PIL): 已安装")
        info.append(f"NumPy版本: {np.__version__}")
        
        return True, "\n".join(info)
        
    except Exception as e:
        return False, f"错误: {str(e)}"

def test_utils() -> Tuple[bool, str]:
    """测试工具库"""
    try:
        import pandas as pd
        import requests
        import matplotlib
        
        info = []
        info.append(f"Pandas版本: {pd.__version__}")
        info.append(f"Requests版本: {requests.__version__}")
        info.append(f"Matplotlib版本: {matplotlib.__version__}")
        
        return True, "\n".join(info)
        
    except Exception as e:
        return False, f"错误: {str(e)}"

def main():
    """主测试函数"""
    print_header("视频检索系统 - 环境验证")
    
    print("\n🔍 开始检查环境配置...")
    
    # 定义所有测试
    tests = [
        ("PyTorch + CUDA", test_pytorch),
        ("CLIP模型", test_clip),
        ("YOLOv8", test_yolov8),
        ("Faiss向量库", test_faiss),
        ("FFmpeg", test_ffmpeg),
        ("Flask框架", test_flask),
        ("数据库驱动", test_database),
        ("OpenCV图像处理", test_opencv),
        ("工具库", test_utils),
    ]
    
    results = []
    
    for i, (name, test_func) in enumerate(tests, 1):
        print_test(name, i, len(tests))
        
        try:
            success, info = test_func()
            print(info)
            
            if success:
                print("✅ 测试通过")
                results.append((name, True))
            else:
                print("❌ 测试失败")
                results.append((name, False))
                
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append((name, False))
    
    # 汇总结果
    print_header("测试结果汇总")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    print("\n" + "="*70)
    print(f"总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 恭喜！所有环境配置完美！")
        print("✅ 可以开始开发视频检索系统了！")
    elif passed >= total * 0.8:
        print(f"\n⚠️  大部分环境正常，还有 {total - passed} 项需要修复")
    else:
        print(f"\n❌ 环境配置存在问题，请修复 {total - passed} 项失败的测试")
    
    print("="*70 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)