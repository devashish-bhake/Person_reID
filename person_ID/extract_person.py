import cv2
from ultralytics import YOLO
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

file = "person_ID/test_2.mp4"

model = YOLO("/home/devashish/Desktop/work/Person_ReID/person_ID/personID.pt")
print('model loaded')

def extract_people():
    video = cv2.VideoCapture(file)
    while True:
        ret, frame = video.read()  # Read the video frames
        # Exit the loop if no more frames in either video
        if not ret:
            break
        # frame = cv2.resize(frame, (2960, 1440))
        results = model.predict(frame, save_crop=True, classes = 0)
        res_plotted = results[0].plot()
        cv2.imshow("Tracking_Stream", res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

extract_people()
