from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time

app = Flask(__name__)

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle
def test_camera_fps(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"相機的預設幀數: {fps}")

    actual_fps = 0
    start_time = cv2.getTickCount()

    for i in range(60):  # 測試60幀
        ret, frame = cap.read()
        if not ret:
            print("無法從相機讀取畫面。")
            break

        actual_fps += 1

    end_time = cv2.getTickCount()
    elapsed_time = (end_time - start_time) / cv2.getTickFrequency()
    actual_fps = actual_fps / elapsed_time

    print(f"實際測試的幀數: {actual_fps:.2f} FPS")
    return actual_fps

def generate_frames():
    actual_fps = test_camera_fps(camera_index=0)
    print(f"Actual FPS: {actual_fps}")
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    frame_count = 0
    consecutive_detect = 0
    counter = 0 
    stage = None
    capture_frequency = 5
    capture_interval = 1 / capture_frequency
    start_time = time.time()
    angle_values = []
    foot_angle_values = []
    T_F = None
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            print(f"Read frame successful: {ret}")
            frame_count += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            try:
                landmarks = results.pose_landmarks.landmark
                HIP = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                KNEE = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ANKLE = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                

                angle = calculate_angle(HIP, KNEE, ANKLE)
                
                cv2.putText(image, str(angle), tuple(np.multiply(KNEE, [640, 480]).astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if time.time() - start_time >= capture_interval:
                    angle_values.append(angle)

                if len(angle_values) == capture_frequency:
                    average_angle = sum(angle_values) / len(angle_values)
                    angle_values = []
                
                if average_angle <= 120:
                    stage = "down"
                    if average_angle < 90:
                        T_F = "KNEE Wrong"
                    else:
                        T_F = "correct"
                if average_angle > 160 and stage == 'down':
                    stage = "up"
                    T_F = None
                    counter += 1
                    print(counter)
            
            except:
                pass

            hold_time = 10
            if T_F == "correct":
                hold_time = ''
            count_time = 10
            if T_F == "correct" and count_time > 0:
                consecutive_detect += 1
                count_time = int(count_time - (consecutive_detect / (actual_fps / 2)))
            else:
                consecutive_detect = 0
            
            if consecutive_detect == 90:
                print("SUCCESS")

            cv2.rectangle(image, (0, 0), (1000, 73), (1, 180, 104), -1)
            cv2.putText(image, 'REPS', (15, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'STAGE', (220, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (215, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'T/F', (550, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, T_F, (485, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, 'TIME', (900, 12), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)  
            cv2.putText(image, str(hold_time), (895, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            if T_F == "correct" and count_time > 0:
                cv2.putText(image, str(count_time), (895, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            if T_F == "correct" and count_time <= 0:
                cv2.putText(image, 'upup', (895, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))               
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    return render_template('index.html')  # render the HTML page

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
