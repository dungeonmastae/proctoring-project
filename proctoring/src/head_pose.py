from audioop import avg
from glob import glob
from itertools import count
import cv2
import mediapipe as mp
import numpy as np
import threading as th
import sounddevice as sd
import pyaudio

CHUNK = 1024  # Number of frames per buffer
FORMAT = pyaudio.paInt16  # Audio format (16-bit int)
CHANNELS = 1  # Single channel (mono)
RATE = 44100  # Sampling rate (samples/second)

# place holders and global variables
x = 0                                       # X axis head pose
y = 0                                       # Y axis head pose

X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0




def pose():
    global VOLUME_NORM, x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT
    #############################
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    # mp_drawing_styles = mp.solutions
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    phone_cascade = cv2.CascadeClassifier("phone_cascade.xml")  # Path to phone cascade classifier
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open microphone stream
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

    def print_sound(indata, frames, time, status):
        global SOUND_AMPLITUDE
        vnorm = np.linalg.norm(indata) * 10
        SOUND_AMPLITUDE = vnorm
        # Detect noise based on a threshold
        # if vnorm > SOUND_AMPLITUDE_THRESHOLD:
        #     print("Noise detected!")
        # Set up the audio stream callback

    stream_callback = print_sound

    while cap.isOpened():

        data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

        # Calculate root mean square (RMS) value
        rms = np.sqrt(np.mean(np.square(data.astype(np.float64))))

        # Detect noise based on a threshold
        threshold = 500  # Adjust this value according to your environment
        if rms > threshold:
            text = "Noise Detected! AND Looking"
            print("Noise detected!")
            X_AXIS_CHEAT = 1
            Y_AXIS_CHEAT = 1
        else:
            text="Looking"
            X_AXIS_CHEAT = 0

        success, image = cap.read()

        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #
        # phones = phone_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=15, minSize=(30, 30))
        #
        # if len(phones) > 0:
        #     print("Phone detected")
        faces = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        if len(faces) == 1:
            print("Face detected")
            X_AXIS_CHEAT = 0
            Y_AXIS_CHEAT = 0
        elif len(faces)==0 or len(faces)>1:
            print("No face detected")
            X_AXIS_CHEAT = 1
            Y_AXIS_CHEAT=1
        # To improve performance

        image.flags.writeable = False

        # Get the result
        results = face_mesh.process(image)
        
        # To improve performance
        image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        
        face_ids = [33, 263, 1, 61, 291, 199]

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None)
                for idx, lm in enumerate(face_landmarks.landmark):
                    # print(lm)
                    if idx in face_ids:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360

                # print(y)

                # See where the user's head tilting
                if y < -10:
                    text += "Looking Left"
                elif y > 10:
                    text += "Looking Right"
                elif x < -10:
                    text += "Looking Down"
                else:
                    text += "Forward"
                text = str(int(x)) + "::" + str(int(y)) + text
                # print(str(int(x)) + "::" + str(int(y)))
                # print("x: {x}   |   y: {y}  |   sound amplitude: {amp}".format(x=int(x), y=int(y), amp=audio.SOUND_AMPLITUDE))
                
                # Y is left / right
                # X is up / down
                if y < -10 or y > 10:
                    X_AXIS_CHEAT = 1
                else:
                    X_AXIS_CHEAT = 0

                if x < -5:
                    Y_AXIS_CHEAT = 1
                else:
                    Y_AXIS_CHEAT = 0

                # print(X_AXIS_CHEAT, Y_AXIS_CHEAT)
                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                
                cv2.line(image, p1, p2, (255, 0, 0), 2)

                # Add the text on the image
                cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

#############################
if __name__ == "__main__":
    t1 = th.Thread(target=pose)

    t1.start()

    t1.join()