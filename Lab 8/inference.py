import os
import cv2
import argparse
from pathlib import Path
import config
from models import FaceEmotionSystem
from utils import visualize_results_grid


def process_image(
    image_path: str,
    output_path: str = None,
    show_results: bool = True,
    save_results: bool = True,
):

    system = FaceEmotionSystem(config.Config.MODEL_SAVE_PATH)

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to read image at {image_path}")
        return

    print("Detecting faces and classifying emotions...")
    boxes, emotions, confidences = system.detect_and_classify(image)

    print(f"\nFound {len(boxes)} face(s):")
    for i, (box, emotion, confidence) in enumerate(zip(boxes, emotions, confidences)):
        x1, y1, x2, y2 = box
        print(f"Face {i+1}:")
        print(f"  Position: ({x1}, {y1}) to ({x2}, {y2})")
        print(f"  Emotion: {emotion}")
        print(f"  Confidence: {confidence:.3f}")
        print()

    result_image = system.draw_results(image, boxes, emotions, confidences)

    if save_results:
        if output_path is None:

            input_path = Path(image_path)
            output_path = str(
                input_path.parent / f"{input_path.stem}_result{input_path.suffix}"
            )

        cv2.imwrite(output_path, result_image)
        print(f"Result saved to: {output_path}")

    if show_results:

        grid_image = visualize_results_grid(image, boxes, emotions, confidences)

        cv2.imshow("Face Emotion Detection", grid_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return boxes, emotions, confidences


def process_video(
    video_path: str,
    output_path: str = None,
    max_frames: int = 100,
    show_results: bool = True,
):

    system = FaceEmotionSystem(config.Config.MODEL_SAVE_PATH)

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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

        if frame_count % 5 == 0:

            boxes, emotions, confidences = system.detect_and_classify(frame)

            result_frame = system.draw_results(frame, boxes, emotions, confidences)

            cv2.putText(
                result_frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                result_frame,
                f"Faces: {len(boxes)}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            processed_count += 1

            if output_path:
                out.write(result_frame)

            if show_results:
                cv2.imshow("Face Emotion Detection - Video", result_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

    print(f"\nVideo processing completed.")
    print(f"Total frames: {frame_count}, Processed frames: {processed_count}")
    if output_path:
        print(f"Output saved to: {output_path}")


def process_webcam():

    system = FaceEmotionSystem(config.Config.MODEL_SAVE_PATH)

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

        boxes, emotions, confidences = system.detect_and_classify(frame)

        result_frame = system.draw_results(frame, boxes, emotions, confidences)

        cv2.putText(
            result_frame,
            "Face Emotion Detection - Webcam",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            result_frame,
            f"Faces detected: {len(boxes)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            result_frame,
            "Press 'q' to quit",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        cv2.imshow("Face Emotion Detection - Webcam", result_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):

            import time

            timestamp = int(time.time())
            filename = f"webcam_capture_{timestamp}.jpg"
            cv2.imwrite(filename, result_frame)
            print(f"Frame saved as: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nWebcam processing stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Face Emotion Detection and Classification"
    )
    parser.add_argument("--input", type=str, help="Input image/video path")
    parser.add_argument("--output", type=str, help="Output path")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["image", "video", "webcam"],
        default="image",
        help="Processing mode",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=100,
        help="Maximum frames to process for video",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not show results")
    parser.add_argument("--no-save", action="store_true", help="Do not save results")

    args = parser.parse_args()

    if args.mode == "image":
        if not args.input:
            parser.error("--input is required for image mode")
        process_image(
            args.input,
            args.output,
            show_results=not args.no_show,
            save_results=not args.no_save,
        )
    elif args.mode == "video":
        if not args.input:
            parser.error("--input is required for video mode")
        process_video(
            args.input,
            args.output,
            max_frames=args.max_frames,
            show_results=not args.no_show,
        )
    elif args.mode == "webcam":
        process_webcam()


if __name__ == "__main__":
    main()
