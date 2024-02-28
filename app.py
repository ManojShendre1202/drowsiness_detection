import streamlit as st
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
from detector import *

thresh = st.number_input("Threshold", placeholder="Threshold for closed eyes")
frame_check = st.number_input("Frame Check", placeholder = "Min number of frame to check for the closed eyes",step = 1)
min_open_frames = st.number_input("Min frames for Video", placeholder="Enter the min frames that mouth should be opened", step = 1) 
sustained_yawn_threshold = st.number_input("Minimum number of sustained yawns to trigger alert", placeholder="Minimum number of sustained yawns to trigger alert", step = 1)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

(l_eye_start, l_eye_end) = (42, 48)
(r_eye_start, r_eye_end) = (36, 42)
(mouth_start, mouth_end) = (48, 68)

counter = 0
alarm_playing = False
yawning = False
max_closed_frames = 5  


# Variables to track yawning state
yawn_frames = 0
yawn_count = 0
sustained_yawn_count = 0
previous_mouth_closed = True 



tb1, tb2 = st.tabs(["Live web cam","Settings"])
with tb1:
    st.title("Webcam Live Feed")
    run = st.checkbox('Run', key = "Live feed")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        _, frame = camera.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        for face in faces:
            shape = predictor(gray, face)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
            l_eye = shape[l_eye_start:l_eye_end]
            r_eye = shape[r_eye_start:r_eye_end]
            mouth = shape[mouth_start:mouth_end]

            l_ear = calculate_ear(l_eye)
            r_ear = calculate_ear(r_eye)
            ear = (l_ear + r_ear) / 2.0

            # Drowsiness detection logic (unchanged)
            if ear < thresh:
                counter += 1
                if counter >= frame_check and not alarm_playing:
                    cv2.putText(frame, "ALERT! Closing eyes Detected!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    play_alert()
                    alarm_playing = True
                    
            else:
                counter = 0
                alarm_playing = False

            mouth_ratio = calculate_mouth_aspect_ratio(mouth)

            # Sustained yawning detection
            if mouth_ratio > 0.6:
                yawn_frames += 1

                if previous_mouth_closed:
                    yawn_frames = 1
                    previous_mouth_closed = False

                elif yawn_frames >= min_open_frames:
                    # Mouth has been open for a sustained period
                    yawn_count += 1
                    yawn_frames = 0
            else:  # Mouth closed, potentially between yawns
                if yawn_frames > 0:  # Mouth was previously open
                    if yawn_frames <= max_closed_frames:
                        # Brief closure considered part of the yawn
                        yawn_frames += 1
                    else:
                        # Mouth closed for too long, reset yawn sequence
                        yawn_frames = 0
                        yawn_count = 0
                previous_mouth_closed = True

            if yawn_count >= sustained_yawn_threshold and not alarm_playing:
                cv2.putText(frame, "ALERT! Excessive Yawning Detected!", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                play_alert()
                sustained_yawn_count += 1
                alarm_playing = True
                

            # Reset counters after alert is played
            if sustained_yawn_count > 0 and not alarm_playing:
                sustained_yawn_count = 0

            # Draw facial landmarks and yawning count
            cv2.drawContours(frame, [cv2.convexHull(np.array(mouth))], -1, (0, 255, 0), 1)
            cv2.putText(frame, f"Yawning Count: {yawn_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.drawContours(frame, [cv2.convexHull(np.array(l_eye))], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(np.array(r_eye))], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(np.array(mouth))], -1, (0, 255, 0), 1)

        FRAME_WINDOW.image(frame)
    else:
        st.write('Stopped')

with tb2:
    st.write("Settings")

