"""
推理脚本 - 对单张图片进行人脸检测和表情分类
"""
import os
import cv2
import argparse
from pathlib import Path
import config
from models import FaceEmotionSystem
from utils import visualize_results_grid

def process_image(image_path: str, output_path: str = None, 
                  show_results: bool = True, save_results: bool = True):
    """
    处理单张图片
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        show_results: 是否显示结果
        save_results: 是否保存结果
    """
    # 初始化系统
    system = FaceEmotionSystem(config.Config.MODEL_SAVE_PATH)
    
    # 读取图片
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to read image at {image_path}")
        return
    
    # 检测人脸和表情
    print("Detecting faces and classifying emotions...")
    boxes, emotions, confidences = system.detect_and_classify(image)
    
    # 打印结果
    print(f"\nFound {len(boxes)} face(s):")
    for i, (box, emotion, confidence) in enumerate(zip(boxes, emotions, confidences)):
        x1, y1, x2, y2 = box
        print(f"Face {i+1}:")
        print(f"  Position: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Emotion: {emotion}")
        print(f"  Confidence: {confidence:.3f}")
        print()
    
    # 绘制结果
    result_image = system.draw_results(image, boxes, emotions, confidences)
    
    # 保存结果
    if save_results:
        if output_path is None:
            # 自动生成输出路径
            input_path = Path(image_path)
            output_path = str(input_path.parent / f"{input_path.stem}_result{input_path.suffix}")
        
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to: {output_path}")
    
    # 显示结果
    if show_results:
        # 创建网格可视化
        grid_image = visualize_results_grid(image, boxes, emotions, confidences)
        
        # 显示
        cv2.imshow('Face Emotion Detection', grid_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return boxes, emotions, confidences


def process_video(video_path: str, output_path: str = None, 
                  max_frames: int = 100, show_results: bool = True):
    """
    处理视频文件
    
    Args:
        video_path: 视频文件路径
        output_path: 输出视频路径
        max_frames: 最大处理帧数
        show_results: 是否显示结果
    """
    # 初始化系统
    system = FaceEmotionSystem(config.Config.MODEL_SAVE_PATH)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video at {video_path}")
        return
    
    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建视频写入器
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    processed_count = 0
    
    print(f"Processing video: {video_path}")
    print(f"FPS: {fps}, Resolution: {width}x{height}")
    
    while True:
        ret, frame = cap.read()
        if not ret or processed_count >= max_frames:
            break
        
        frame_count += 1
        
        # 每n帧处理一次
        if frame_count % 5 == 0:  # 每5帧处理一次以提高速度
            # 检测人脸和表情
            boxes, emotions, confidences = system.detect_and_classify(frame)
            
            # 绘制结果
            result_frame = system.draw_results(frame, boxes, emotions, confidences)
            
            # 在左上角显示帧信息
            cv2.putText(result_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Faces: {len(boxes)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            processed_count += 1
            
            # 写入输出视频
            if output_path:
                out.write(result_frame)
            
            # 显示结果
            if show_results:
                cv2.imshow('Face Emotion Detection - Video', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    # 释放资源
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nVideo processing completed.")
    print(f"Total frames: {frame_count}, Processed frames: {processed_count}")
    if output_path:
        print(f"Output saved to: {output_path}")


def process_webcam():
    """处理摄像头实时视频流"""
    # 初始化系统
    system = FaceEmotionSystem(config.Config.MODEL_SAVE_PATH)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Failed to open webcam")
        return
    
    print("Starting webcam face emotion detection...")
    print("Press 'q' to quit, 's' to save current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测人脸和表情
        boxes, emotions, confidences = system.detect_and_classify(frame)
        
        # 绘制结果
        result_frame = system.draw_results(frame, boxes, emotions, confidences)
        
        # 显示帧率等信息
        cv2.putText(result_frame, "Face Emotion Detection - Webcam", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, f"Faces detected: {len(boxes)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(result_frame, "Press 'q' to quit", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示结果
        cv2.imshow('Face Emotion Detection - Webcam', result_frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前帧
            import time
            timestamp = int(time.time())
            filename = f"webcam_capture_{timestamp}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"Frame saved as: {filename}")
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam processing stopped.")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Face Emotion Detection and Classification')
    parser.add_argument('--input', type=str, help='Input image/video path')
    parser.add_argument('--output', type=str, help='Output path')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'webcam'], 
                       default='image', help='Processing mode')
    parser.add_argument('--max-frames', type=int, default=100, 
                       help='Maximum frames to process for video')
    parser.add_argument('--no-show', action='store_true', 
                       help='Do not show results')
    parser.add_argument('--no-save', action='store_true', 
                       help='Do not save results')
    
    args = parser.parse_args()
    
    if args.mode == 'image':
        if not args.input:
            parser.error("--input is required for image mode")
        process_image(
            args.input, 
            args.output, 
            show_results=not args.no_show,
            save_results=not args.no_save
        )
    elif args.mode == 'video':
        if not args.input:
            parser.error("--input is required for video mode")
        process_video(
            args.input,
            args.output,
            max_frames=args.max_frames,
            show_results=not args.no_show
        )
    elif args.mode == 'webcam':
        process_webcam()


if __name__ == '__main__':
    main()