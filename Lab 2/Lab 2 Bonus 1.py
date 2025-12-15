import cv2
import os

# Load pre-trained cascade files
face_cascade = cv2.CascadeClassifier("haar-cascade-files/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haar-cascade-files/haarcascade_eye.xml")

# Read image
image = cv2.imread("img/demo.png")
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
)
print("Detected faces:", len(faces))

# Process each face
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # ROI for face
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    
    # OPTIMIZATION 1: Restrict eye search to upper half of face
    face_height_ratio = 0.6
    roi_upper = roi_gray[0:int(h*face_height_ratio), 0:w]
    
    # OPTIMIZATION 2: Adjust detection parameters
    eyes = eye_cascade.detectMultiScale(
        roi_upper, 
        scaleFactor=1.05, 
        minNeighbors=8,
        minSize=(20, 20), 
        maxSize=(60, 60)
    )
    
    print(f"Eyes in face at ({x},{y}): {len(eyes)}")
    
    for (ex, ey, ew, eh) in eyes:
        center_x = x + ex + ew // 2
        center_y = y + ey + eh // 2
        
        radius = min(ew, eh) // 2
        
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 255), 2)

        cv2.drawMarker(image, (center_x, center_y), (255, 255, 255), 
                      markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)

cv2.imshow('Optimized Eye Detection', image)
cv2.imwrite("optimized_eye_detection.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()