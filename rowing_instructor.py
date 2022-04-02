from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow import keras
import cv2
import mediapipe as mp
import numpy as np
import sys


mp_drawing = mp.solutions.drawing_utils
# We want the pose estimation
mp_pose = mp.solutions.pose
POSITION = "left" #"right"
# This values might be needed to tweak
min_legs_angle_threshold = 130
min_arms_angle_threshold = 140
min_detection_confidence = 0.5
min_tracking_confidence = 0.5
min_visibility_values = 0.4
# Perform a good rowing tecnique and test values 
test_values = False
warning_threshold = 2

model = load_model("models/mobile.h5")


def three_joint_angle(a, b, c):
    """
    This function takes 3 points and form an angle over those.
    Arctan can convert x,y plane into an angle and a radius: arctan(y/x).
    Necessary to change radians to angles
    Parameters
    ----------
    a : int
        First point 
    b : int
        Second point
    c : int
        Third point
        
    Returns
    -------
    float
        Angle of the positions
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180:
        angle = 360 - angle
    
    return(angle)


def bad_back_posture(count,
                     cv_image,
                     model, 
                     right_shoulder, 
                     right_hip, 
                     left_shoulder, 
                     left_hip, 
                     nose, 
                     y_val=720, 
                     x_val=1280, 
                     max_x_ease=0.2, 
                     min_x_ease=0.2, 
                     min_y_ease=0.1, 
                     max_y_ease=0.2, 
                     cropped_size=(224, 224), 
                     test_values=False):
    if (count % 10 == 0):
        max_x = max(right_shoulder.x, left_shoulder.x, right_hip.x, left_hip.x)
        max_x = max_x + (max_x * max_x_ease)
        min_x = min(right_shoulder.x, left_shoulder.x, right_hip.x, left_hip.x)
        min_x = min_x - (min_x * min_x_ease)
        min_y = min(right_hip.y, left_hip.y)
        min_y = min_y + (min_y * min_y_ease)
        max_y = nose.y - (nose.y * max_y_ease)
        cropped_img = cv_image[int(max_y*y_val):int(min_y*y_val), int(min_x*x_val):int(max_x*x_val)]
        image_cnn = cv2.resize(cropped_img, cropped_size)
        if test_values:
            return(image_cnn)
        img_array = image.img_to_array(image_cnn)
        img_batch = np.expand_dims(img_array, axis=0)
        processed_image = tf.keras.applications.mobilenet.preprocess_input(img_batch) 
        prediction = model.predict(x=processed_image)[0][1]
        if prediction >= 0.5:
            print("Warning back problems")
            return(1)
        else:
            return(0)


# Probably need to do improvements on how we manage this
# Webcam
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('videos/rowing2.mp4')

# legs straight: 1, otherwise 0.
legs_track = []
# arms straight: 1, otherwise 0.
arms_track = []
warning_technique = 0
warning_back = 0
count = 0

with mp_pose.Pose(min_detection_confidence=min_detection_confidence, 
                  min_tracking_confidence=min_tracking_confidence) as pose:  
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Preprocess data for pose input
        cv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv_image.flags.writeable = False
                
        results = pose.process(cv_image)
        
        cv_image.flags.writeable = True
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            right_shoulder = landmarks[12]
            right_hip = landmarks[24]
            left_shoulder = landmarks[11]
            left_hip = landmarks[23]
            nose = landmarks[0]
            if test_values:
                cv_image = bad_back_posture(count, 
                                            cv_image, 
                                            model, 
                                            right_shoulder, 
                                            right_hip, 
                                            left_shoulder, 
                                            left_hip, 
                                            nose, 
                                            y_val=337, 
                                            x_val=599, 
                                            test_values=True)
            else:
                warning_back = bad_back_posture(count, 
                                                cv_image, 
                                                model, 
                                                right_shoulder, 
                                                right_hip, 
                                                left_shoulder, 
                                                left_hip, 
                                                nose, 
                                                y_val=337, 
                                                x_val=599)
            
            if POSITION == "left":
                # Extract left part of the body
                hip = landmarks[23]
                knee = landmarks[25]
                ankle = landmarks[27]
                shoulder = landmarks[11]
                elbow = landmarks[13]
                wrist = landmarks[15]
            else:
                # Extract right part of the body
                hip = landmarks[24]
                knee = landmarks[26]
                ankle = landmarks[28]
                shoulder = landmarks[12]
                elbow = landmarks[14]
                wrist = landmarks[16]
                
            hip_visibility = hip.visibility
            knee_visibility = knee.visibility
            ankle_visibility = ankle.visibility
            shoulder_visibility = shoulder.visibility
            elbow_visibility = elbow.visibility
            wrist_visibility = wrist.visibility
                        
            if (hip_visibility >= min_visibility_values and 
                knee_visibility >= min_visibility_values and 
                ankle_visibility >= min_visibility_values and
                shoulder_visibility >= min_visibility_values and 
                elbow_visibility >= min_visibility_values and 
                wrist_visibility >= min_visibility_values):
                # Only calculate if visbility
                hip = [hip.x, hip.y]
                knee = [knee.x, knee.y]
                ankle = [ankle.x, ankle.y]
                shoulder = [shoulder.x, shoulder.y]
                elbow = [elbow.x, elbow.y]
                wrist = [wrist.x, wrist.y]

                # Extract angles
                angle_leg = three_joint_angle(hip, knee, ankle)
                angle_arm = three_joint_angle(shoulder, elbow, wrist)
                if test_values:
                    print("Angle of leg: ", angle_leg)
                    print("Angle of arm: ", angle_arm)
                
                if angle_leg >= min_legs_angle_threshold:
                    # legs straight
                    legs_track.append(1)
                else:
                    legs_track.append(0)
                
                if angle_arm >= min_arms_angle_threshold:
                    # arms straight
                    arms_track.append(1)
                else:
                    arms_track.append(0)
            
                # We need information to make decision, at least 3 steps
                if len(legs_track) > 2:
                    if legs_track[-1] != legs_track[-2] and legs_track[-1] != legs_track[-3]:
                        # Would keep track of instant transition
                        # A stroke would be completed by 2 transitions, going backward and then forward.
                        # print("transition")
                        legs_track = []
                        warning_technique = 0
                    else:
                        # If legs not straight and arms not straight: Warning
                        if len(arms_track) > 2:
                            if arms_track[-1] != arms_track[-2] and arms_track[-1] != arms_track[-3]:
                                # Would keep track of instant transition
                                # A stroke would be completed by 2 transitions, going backward and then forward.
                                # print("transition")
                                arms_track = []
                                warning_technique = 0
                            elif legs_track[-1] == 0 and arms_track[-1] == 0:
                                warning_technique += 1
                                # If legs not straight and arm not straight then warning

                    # Only when legs straight you should pull elbows back
                    # 0 and 1 legs not straight and arms straight
                    # 1 and 0 legs straight and arms not straight
                    # 1 and 1 legs straight and arms straight (should be instantaneous)
                    # 0 and 0 legs not straight and arms not straight warning

        except:
            pass
        
        count += 1
        
        if not test_values:
            if (warning_technique > warning_threshold) or (warning_back == 1):
                sys.stdout.write('\a')
                sys.stdout.flush()
                cv2.rectangle(cv_image,
                              (0, 0), 
                              (130, 40), 
                              (0, 0, 255),
                              -1)
                cv2.putText(cv_image,
                            'Warning', 
                            (2, 27), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (255, 255, 255), 
                            2, 
                            cv2.LINE_AA)

            mp_drawing.draw_landmarks(cv_image, 
                                      results.pose_landmarks, 
                                      mp_pose.POSE_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), 
                                                             thickness=2, 
                                                             circle_radius=2))
        
        try:
            cv2.imshow("mediapipe test", cv_image)
        except:
            pass
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
