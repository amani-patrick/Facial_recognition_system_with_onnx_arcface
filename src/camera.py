# src/camera.py
import cv2
def main():
    camera_capture = cv2.VideoCapture(0)
    if not camera_capture.isOpened():
        raise RuntimeError("Camera not opened. Try changing index (0/1/2).")
    
    print("Camera test. Press 'q' to quit.")
    while True:
        success, current_frame = camera_capture.read()
        if not success:
            print("Failed to read frame.")
            break
        cv2.imshow("Camera Test", current_frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    camera_capture.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()