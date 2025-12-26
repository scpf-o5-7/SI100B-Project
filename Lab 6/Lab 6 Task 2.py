import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import sys
import os

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


class Detector:
    def __init__(self, CascadePath, ModelPath):
        self.classes = ["happy", "neutral", "sad"]

        self.face_cascade = cv2.CascadeClassifier(CascadePath)

        self.model = SI100FaceNet(num_classes=3, printtoggle=False)
        self.model.load_state_dict(torch.load(ModelPath, weights_only=True))
        self.model.eval()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded on device: {self.device}")

    def process(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        print(f"Detected {len(faces)} faces")

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_roi = img[y : y + h, x : x + w]

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

    def transform2tensor(self, face_img):
        face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        tensor_data = transform(face_img_rgb)
        return tensor_data.unsqueeze(0)


if __name__ == "__main__":
    demo_image = cv2.imread("../Lab 2/img/demo.png")

    cascade_path = "../Lab 2/haar-cascade-files/haarcascade_frontalface_default.xml"
    model_path = "../Lab 4/face_expression.pth"

    detector = Detector(cascade_path, model_path)

    result_image = detector.process(demo_image)

    cv2.imwrite("lab6_result.png", result_image)
    cv2.imshow("Face Emotion Detection", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()