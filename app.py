from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import time

app = Flask(__name__)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

def gen_frames():
    cap = cv2.VideoCapture(0)
    prev_time = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(frame_rgb)
            binary_array = [0, 0, 0, 0, 0]
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    # Get the coordinates of the fingertips and the thumb
                    landmarks = hand_landmarks.landmark
                    for idx, tip in enumerate(finger_tips):
                        if landmarks[tip].y < landmarks[tip - 2].y:
                            binary_array[idx + 1] = 1
                    # Check the thumb separately
                    if landmarks[thumb_tip].x > landmarks[thumb_tip - 2].x:
                        binary_array[0] = 1

            current_time = time.time()
            if current_time - prev_time > 3:
                prev_time = current_time
                print(f"Binary Array: {binary_array}")
            
            # Add the binary array to the frame
            array_text = f"Array: {binary_array}"
            cv2.putText(frame, array_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
