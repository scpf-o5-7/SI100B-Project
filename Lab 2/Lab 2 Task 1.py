import cv2

img = cv2.imread("img/create.png")
cv2.rectangle(img, (80, 60), (80 + 480, 60 + 360), (0, 255, 0), 2)
img = img[30:450, 40:600]
img = cv2.resize(img, None, fx=0.5, fy=0.5)
text = "I love SI100B"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
thickness = 2
(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
height, width = img.shape[:2]
x = (width - text_width) // 2
y = height // 2 + text_height // 2
cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness)
cv2.imwrite("result.png", img)
cv2.imshow("src", img)
cv2.waitKey(0)
