import cv2
from ultralytics import YOLO
import multiprocessing
import warnings
warnings.filterwarnings("ignore")


# video inputs
video_1 = "/home/devashish/Desktop/work/Person_ReID/person_ID/test_1.mp4"
video_2 = "/home/devashish/Desktop/work/Person_ReID/person_ID/test_2.mp4"
video_3 = "/home/devashish/Desktop/work/Person_ReID/person_ID/test_3.mp4"
video_4 = "/home/devashish/Desktop/work/Person_ReID/person_ID/test_4.mp4"
# change the above paths to your vwebcam ids if multiple webcams are connected
#example: video_1 = 0, video_2 = 1, video_3 = 2 and so on where 0, 1, 2 ... are the webcam ids of the webcams connected either default or via usb connectors

model = YOLO("/home/devashish/Desktop/work/Person_ReID/person_ID/personID.pt")
print('model loaded')

def run_tracker_in_thread(filename, model, file_index):
    print("video file: ", filename)
    video = cv2.VideoCapture(filename)
    while True:
        ret, frame = video.read()  # Read the video frames
        # Exit the loop if no more frames in either video
        if not ret:
            break
        frame = cv2.resize(frame, (640, 480))
        results = model.predict(frame, classes = 0)
        res_plotted = results[0].plot()
        cv2.imshow("Tracking_Stream_"+str(file_index), res_plotted)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()

tracker_thread1 = multiprocessing.Process(target=run_tracker_in_thread,
                                   args=(video_1, model, 1))

# Thread used for the webcam
tracker_thread2 = multiprocessing.Process(target=run_tracker_in_thread,
                                   args=(video_2, model, 2))



tracker_thread1.start()
tracker_thread2.start()
cv2.destroyAllWindows()