import cv2
import numpy as np


def nms(boxes, scores, threshold=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    areas = boxes[:, 2] * boxes[:, 3]

    indices = np.argsort(scores)[::-1]

    keep = []

    while len(indices) > 0:

        current_index = indices[0]
        keep.append(current_index)

        if len(indices) == 1:
            break

        current_x1 = x1[current_index]
        current_y1 = y1[current_index]
        current_x2 = x2[current_index]
        current_y2 = y2[current_index]

        other_indices = indices[1:]
        other_x1 = x1[other_indices]
        other_y1 = y1[other_indices]
        other_x2 = x2[other_indices]
        other_y2 = y2[other_indices]

        xx1 = np.maximum(current_x1, other_x1)
        yy1 = np.maximum(current_y1, other_y1)
        xx2 = np.minimum(current_x2, other_x2)
        yy2 = np.minimum(current_y2, other_y2)

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        iou = intersection / (
            areas[current_index] + areas[other_indices] - intersection
        )

        indices = other_indices[iou <= threshold]

    return keep


face_cascade = cv2.CascadeClassifier(
    "haar-cascade-files/haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier("haar-cascade-files/haarcascade_eye.xml")

image = cv2.imread("img/demo.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)

for x, y, w, h in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    roi_gray = gray[y : y + h, x : x + w]
    roi_upper = roi_gray[0 : int(h * 0.6), 0:w]

    eyes, _, confidences = eye_cascade.detectMultiScale3(
        roi_upper,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(15, 15),
        maxSize=(60, 60),
        outputRejectLevels=True,
    )

    if len(eyes) > 0:
        if len(confidences) == 0:
            confidences = [ew * eh for (ex, ey, ew, eh) in eyes]

        keep_indices = nms(eyes.tolist(), confidences, threshold=0.3)
        filtered_eyes = eyes[keep_indices]

        for ex, ey, ew, eh in filtered_eyes:
            center_x = x + ex + ew // 2
            center_y = y + ey + eh // 2
            radius = min(ew, eh) // 2
            cv2.circle(image, (center_x, center_y), radius, (0, 0, 255), 2)

            cv2.putText(
                image,
                "NMS",
                (center_x - 15, center_y - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 0),
                1,
            )

cv2.imshow("Eye Detection with NMS", image)
cv2.imwrite("eye_detection_with_nms.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
