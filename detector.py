import cv2
from scipy.spatial import distance as dist
import dlib
import pygame.mixer as mixer
import imutils
import numpy as np

def play_alert():
    mixer.init()
    mixer.music.load('data/music.wav')
    mixer.music.play()

def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])  # 48, 54
    B = dist.euclidean(mouth[0], mouth[6])  # 60, 64
    return A / B
