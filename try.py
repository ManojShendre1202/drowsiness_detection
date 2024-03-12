import cv2
import dlib

# Load face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/datascienceprojects/drowsiness_detection/data/shape_predictor_68_face_landmarks.dat")

# Start video capture 
cap = cv2.VideoCapture(0)

# Yawn detection parameters
YAWN_THRESHOLD = 25 
yawn_count = 0
yawn_status = False 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Mouth edge indices
        mouth_indices = range(48, 68)

        # Extract mouth coordinates
        mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in mouth_indices]

        # Calculate distance between top and bottom lips
        top_lip_center = (mouth_points[13][0] + mouth_points[14][0]) // 2, (mouth_points[13][1] + mouth_points[14][1]) // 2
        bottom_lip_center = (mouth_points[19][0] + mouth_points[18][0]) // 2, (mouth_points[19][1] + mouth_points[18][1]) // 2
        lip_distance = abs(top_lip_center[1] - bottom_lip_center[1])

        # Yawn detection
        if lip_distance > YAWN_THRESHOLD:
            if not yawn_status:  # Check if a yawn wasn't already in progress
                yawn_status = True
                cv2.putText(frame, "Yawn Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                yawn_count += 1

        else:
            yawn_status = False

        # Draw lines around outer mouth edge
        for i in range(len(mouth_points) - 1):
            cv2.line(frame, mouth_points[i], mouth_points[i + 1], (0, 255, 0), 2)

    cv2.imshow('Mouth and Yawn Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
