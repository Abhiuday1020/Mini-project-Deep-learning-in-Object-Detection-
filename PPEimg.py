from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

results = model("Construction_safety/test/images/class1_150_jpg.rf.5995dce34d38deb9eb0b6e36cae78f17.jpg", show=True)

cv2.waitKey(0)
