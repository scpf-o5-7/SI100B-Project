import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import sys

sys.path.append("../Lab 2")
sys.path.append("../Lab 3")
sys.path.append("../Lab 4")

from model import SI100FaceNet

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)


def nms(boxes, scores, overlap_threshold=0.5):
    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = x1 + boxes_np[:, 2]
    y2 = y1 + boxes_np[:, 3]
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    idxs = np.argsort(scores)[::-1]
    
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        
        if len(idxs) == 1:
            break
            
        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])
        
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        overlap = (w * h) / (areas[idxs[1:]] + areas[i] - w * h)
        
        idxs = idxs[1:][overlap < overlap_threshold]
    
    return keep


class Detector:
    _instance = None
    _initialized = False

    def __new__(cls, cascade_path=None, model_path=None):
        if cls._instance is None:
            cls._instance = super(Detector, cls).__new__(cls)
        return cls._instance

    def __init__(self, cascade_path=None, model_path=None):
        if (
            not self._initialized
            and cascade_path is not None
            and model_path is not None
        ):
            self.classes = ["happy", "neutral", "sad"]

            self.face_cascade = cv2.CascadeClassifier(cascade_path)

            self.model = SI100FaceNet(num_classes=3, printtoggle=False)
            self.model.load_state_dict(torch.load(model_path, weights_only=True))
            self.model.eval()

            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

            self._initialized = True

    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.15, 
            minNeighbors=5, 
            minSize=(50, 50), 
            maxSize=(400, 400)
        )

        all_faces = []
        all_confidences = []
        
        for x, y, w, h in faces:
            aspect_ratio = w / h
            if 0.5 < aspect_ratio < 2.0:
                face_roi = img[y:y+h, x:x+w]
                is_face, confidence = self.is_likely_face(face_roi)
                if is_face:
                    all_faces.append((x, y, w, h))
                    all_confidences.append(confidence)
        
        if len(all_faces) > 0:
            keep_indices = nms(all_faces, all_confidences, overlap_threshold=0.5)
            valid_faces = [all_faces[i] for i in keep_indices]
        else:
            valid_faces = []
        
        for x, y, w, h in valid_faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = img[y:y+h, x:x+w]

            if face_roi.size == 0:
                continue

            try:
                tensor_data = self.transform2tensor(face_roi)

                with torch.no_grad():
                    tensor_data = tensor_data.to(self.device)
                    outputs = self.model(tensor_data)
                    _, predicted = torch.max(outputs, 1)
                    emotion = self.classes[predicted.item()]

                label = f"{emotion}"
                cv2.putText(
                    img,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )

                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence = probabilities[0][predicted.item()].item()
                conf_text = f"{confidence:.2f}"
                cv2.putText(
                    img,
                    conf_text,
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1,
                )

            except Exception as e:
                print(f"Error processing face: {e}")
                continue

        return img

    def is_likely_face(self, face_roi):
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        skin_ratio = np.sum(skin_mask > 0) / (face_roi.shape[0] * face_roi.shape[1])
        
        confidence = skin_ratio
        is_face = skin_ratio > 0.15
        
        return is_face, confidence

    def transform2tensor(self, face_img):
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        tensor_data = transform(face_img_rgb)
        return tensor_data.unsqueeze(0)


def process_video(input_video_path, output_video_path, cascade_path, model_path):
    detector = Detector(cascade_path, model_path)

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = detector.process(frame)
        out.write(processed_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cascade_path = "../Lab 2/haar-cascade-files/haarcascade_frontalface_default.xml"
    model_path = "../Lab 4/face_expression.pth"

    input_path = "test.mp4"
    output_path = "processed.mp4"

    process_video(input_path, output_path, cascade_path, model_path)
    print(f"Saved as {output_path}")