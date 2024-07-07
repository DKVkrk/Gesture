import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_module = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def position(frame, results):
    landmark_list = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([id, cx, cy])
    return landmark_list

def recgogesture(landmarks):
    if len(landmarks) == 21:
        thumb_tip = landmarks[4]
        thumb_mcp = landmarks[2]
        index_tip = landmarks[8]

        
        if thumb_tip[2] < thumb_mcp[2] < index_tip[2]:
            return "Thumb Up!"

    return "Not recognized"

def authegesture(gesture):
    correct_gesture = "Thumb Up!"
    return gesture == correct_gesture

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands_module.process(rgb_frame)
        landmark_list = position(frame, results)
        gesture = recgogesture(landmark_list)
        
        if authegesture(gesture):
            cv2.putText(frame, 'Authenticated', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'Not Authenticated', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Gesture Authentication', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break        

    cap.release()
    cv2.destroyAllWindows()
