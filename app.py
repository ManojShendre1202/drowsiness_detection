import cv2
import dlib
import time
import pygame
from scipy.spatial import distance as dist

# Constants
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 20 
TOTAL_BLINK_THRESHOLD = 25
YAWN_THRESHOLD = 30
YAWN_MIN_DURATION = 1.5
YAWN_REPETITION_THRESHOLD = 4

# Global variables
closed_eyes_counter = 0
yawn_counter = 0
yawn_start_time = None

# Load face detector, landmark predictor, and initialize pygame mixer
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/datascienceprojects/drowsiness_detection/data/shape_predictor_68_face_landmarks.dat")
pygame.mixer.init()
pygame.mixer.music.load("C:/datascienceprojects/drowsiness_detection/data/music.wav") 

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# Main function
def main():
    global closed_eyes_counter, yawn_counter, yawn_start_time
    
    # Start video capture 
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            mouth_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            top_lip_center = (mouth_points[13][0] + mouth_points[14][0]) // 2, (mouth_points[13][1] + mouth_points[14][1]) // 2
            bottom_lip_center = (mouth_points[19][0] + mouth_points[18][0]) // 2, (mouth_points[19][1] + mouth_points[18][1]) // 2
            lip_distance = abs(top_lip_center[1] - bottom_lip_center[1])

            # Yawn detection 
            if lip_distance > YAWN_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                if time.time() - yawn_start_time >= YAWN_MIN_DURATION:
                    yawn_counter += 1
                    if yawn_counter >= YAWN_REPETITION_THRESHOLD:
                        cv2.putText(frame, "YAWNING", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        pygame.mixer.music.play()
                        yawn_counter = 0  
                    yawn_start_time = None  
            else:
                yawn_start_time = None
                
            for i in range(len(mouth_points) - 1):
                cv2.line(frame, mouth_points[i], mouth_points[i + 1], (0, 255, 0), 2)

            left_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

            left_ear = eye_aspect_ratio(left_eye_points)
            right_ear = eye_aspect_ratio(right_eye_points)
            ear = (left_ear + right_ear) / 2.0  

            # Eye closure detection
            if ear < EYE_AR_THRESH:
                closed_eyes_counter += 1
                if closed_eyes_counter >= TOTAL_BLINK_THRESHOLD:
                    if closed_eyes_counter >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        pygame.mixer.music.play()  
            else:
                closed_eyes_counter = 0
        for point in left_eye_points:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        for point in right_eye_points:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)
        cv2.imshow('Drowsiness Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
