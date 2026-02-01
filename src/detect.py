import cv2

def main():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")
    
    camera_capture = cv2.VideoCapture(0)
    if not camera_capture.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2")
    
    print("Haar face detect (minimal). Press 'q' to quit.")
    while True:
        success, current_frame = camera_capture.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        detected_faces = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors= 5,
            minSize= (60,60)
        )

        for(x,y,w,h) in detected_faces:
            cv2.rectangle(current_frame, (x,y), (x+w, y+h), (0,255,0),2)

        cv2.imshow("Face Detection", current_frame)
        if(cv2.waitKey(1) & 0xFF== ord("q")):
            break

    camera_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()