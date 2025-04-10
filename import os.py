import cv2
import mediapipe as mp
import pyautogui

mp_hands = mp.solutions.hands
mp_plot = mp.solutions.drawing_utils
handdetect = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

screenwidth, screenheight = pyautogui.size()

cap = cv2.VideoCapture(0)

cv2.namedWindow("Live Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Tracking", 300, 200)

clicking = False

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = handdetect.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            index_finger = hand.landmark[8]  
            thumb_finger = hand.landmark[4] 
           
            x = int(index_finger.x * screenwidth)
            y = int(index_finger.y * screenheight)
            pyautogui.moveTo(x, y, duration=0.1)
            distance = ((index_finger.x - thumb_finger.x) ** 2 + (index_finger.y - thumb_finger.y) ** 2) ** 0.5
            if distance < 0.05:  
                if not clicking: 
                    pyautogui.click()
                    clicking = True
            else:
                clicking = False
            mp_plot.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Live Tracking", frame)
    cv2.moveWindow("Live Tracking", 50, 50) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()