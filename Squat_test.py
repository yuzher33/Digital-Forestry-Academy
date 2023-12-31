import cv2
import mediapipe as mp
import numpy as np
import time
import random

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
    
    return angle

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    cap = cv2.VideoCapture(0)
    frame_count = 0
    consecutive_detect = 0
    # Curl counter variables
    counter = 0 
    stage = None
    # 設定擷取頻率（每秒擷取五次）
    capture_frequency = 5
    capture_interval = 1 / capture_frequency
    start_time = time.time()
    # 初始化變數
    angle_values = []
    foot_angle_values = []
    T_F=None
    
    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            # Frame capture
            frame_count += 1
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates
                HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ANKLE = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                HEEL = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].x,landmarks[mp_pose.PoseLandmark.LEFT_HEEL.value].y]
                FOOTINDEX = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y]
                # Calculate angle
                angle = calculate_angle(HIP, KNEE, ANKLE)
                foot_angle = calculate_angle(KNEE,HEEL,FOOTINDEX)
                # Visualize angle
                cv2.putText(image, str(angle), 
                            tuple(np.multiply(KNEE, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                    )
                # 擷取五次角度值
                if time.time() - start_time >= capture_interval:
                    angle_values.append(angle)  # 假設有一個取得角度的函數
                if time.time() - start_time >= capture_interval:
                    foot_angle_values.append(foot_angle)

                # 擷取五次後計算平均值
                if len(angle_values) == capture_frequency:
                    average_angle = sum(angle_values) / len(angle_values)
                    print("Average Angle:", average_angle)
                    angle_values = []  # 清空列表，準備下一輪擷取
                if len(foot_angle_values) == capture_frequency:
                    foot_average_angle = sum(foot_angle_values) / len(foot_angle_values)
                    print("foot Average Angle:", foot_average_angle)
                    foot_angle_values = []
                # Curl counter logic
                if average_angle <= 120:
                    stage = "down"
                    if average_angle<90:
                        #T_F=None
                        T_F = "KNEE Wrong"
                    else :
                        #T_F=None
                        T_F="correct"
                if average_angle > 160 and stage =='down':
                    stage="up"
                    T_F=None
                    counter +=1
                    print(counter)
            

            except:
                pass
            hold_time = 3
            if T_F =="correct":
                hold_time = ''
            count_time = 3
            if T_F == "correct" and count_time > 0:
                consecutive_detect += 1
                count_time = int(count_time - (consecutive_detect/30))
            
            else:
                consecutive_detect = 0
            if consecutive_detect == 90:
                print("SUCCESS")
            

            # Render curl counter
            # 框框設定
            
            cv2.rectangle(image, (0,0), (1000,73), (1,180,104), -1)
            
            # Rep data
            cv2.putText(image, 'REPS', (15,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), 
                        (10,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            
            # Stage data
            cv2.putText(image, 'STAGE', (220,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, 
                        (215,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            # T/F
            cv2.putText(image, 'T/F', (550,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, T_F, 
                        (485,60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            # time
            
            cv2.putText(image, 'TIME', (900,12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)  
            cv2.putText(image, str(hold_time), 
                                (895,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            if T_F == "correct" and count_time > 0:
                cv2.putText(image, str(count_time), 
                                (895,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            if T_F == "correct" and count_time <= 0:
                cv2.putText(image, 'upup', 
                                (895,60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
                #break
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

main()

