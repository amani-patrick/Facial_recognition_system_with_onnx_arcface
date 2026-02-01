"""
Minimal pipeline:
camera -> Haar face box -> MediaPipe FaceMesh (full-frame) -> extract 5 keypoints -> draw

Run:
python -m src.landmarks

Keys:
q : quit
"""
import cv2
import numpy as np
import mediapipe as mp
# 5-point indices (FaceMesh)
IDX_LEFT_EYE = 33
IDX_RIGHT_EYE = 263
IDX_NOSE_TIP = 1
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291
def main():
# Haar
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load cascade: {cascade_path}")
    # FaceMesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    camera_capture = cv2.VideoCapture(0)
    if not camera_capture.isOpened():
        raise RuntimeError("Camera not opened. Try camera index 0/1/2.")
    
    print("Haar + FaceMesh 5pt (minimal). Press 'q' to quit.")
    while True:
        success, current_frame = camera_capture.read()
        if not success:
            break
        height, width = current_frame.shape[:2]
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        # draw ALL haar faces (no ranking)
        for (x, y, w, h) in detected_faces:
            cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # FaceMesh on full frame (simple)
        rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        face_mesh_result = face_mesh.process(rgb_frame)

        if face_mesh_result.multi_face_landmarks:
            landmarks = face_mesh_result.multi_face_landmarks[0].landmark
            keypoint_indices = [IDX_LEFT_EYE, IDX_RIGHT_EYE, IDX_NOSE_TIP, IDX_MOUTH_LEFT, IDX_MOUTH_RIGHT]
            
            points = []
            for i in keypoint_indices:
                p = landmarks[i]
                points.append([p.x * width, p.y * height])
            keypoints = np.array(points, dtype=np.float32) # (5,2)

            # enforce left/right ordering
            if keypoints[0, 0] > keypoints[1, 0]:
                keypoints[[0, 1]] = keypoints[[1, 0]]
            if keypoints[3, 0] > keypoints[4, 0]:
                keypoints[[3, 4]] = keypoints[[4, 3]]
            
            # draw 5 points
            for (point_x, point_y) in keypoints.astype(int):
                cv2.circle(current_frame, (int(point_x), int(point_y)), 4, (0, 255, 0), -1)
            
            cv2.putText(current_frame, "5pt", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("5pt Landmarks", current_frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break


    camera_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()