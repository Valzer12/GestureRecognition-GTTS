import cv2
import mediapipe as mp
from gtts import gTTS
import pygame
import os
import time

# ==== Inisialisasi pygame untuk audio ====
pygame.mixer.init()

# ==== Inisialisasi MediaPipe ====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_style = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==== Kamera ====
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ==== Pesan per gesture ====
gesture_messages = {
    "ONE": "Perkenalkan",
    "TWO": "Nama Saya,Andhika",
    "FIVE": " Hallo",
    "FIST": "Salam Kenal",
    "THUMB": "Terima Kasih"
}

# ==== Preload semua suara ====
sound_files = {}
os.makedirs("voices", exist_ok=True)

for gesture, text in gesture_messages.items():
    file_path = f"voices/{gesture}.mp3"
    if not os.path.exists(file_path):
        tts = gTTS(text=text, lang="id")
        tts.save(file_path)
    sound_files[gesture] = file_path

# ==== Fungsi untuk memainkan suara ====
def play_voice(gesture):
    try:
        pygame.mixer.music.load(sound_files[gesture])
        pygame.mixer.music.play()
    except Exception as e:
        print("Error saat memutar suara:", e)

# ==== Fungsi deteksi gesture ====
def detect_gesture(hand_landmarks, hand_label):
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]
    thumb_tip = 4
    thumb_mcp = 2
    landmarks = hand_landmarks.landmark

    # Deteksi jari telunjuk - kelingking
    fingers = [landmarks[tip].y < landmarks[mcp].y - 0.02 for tip, mcp in zip(finger_tips, finger_mcp)]

    # Deteksi ibu jari terbuka ke arah luar (horizontal)
    if hand_label == "Right":
        thumb_open_side = landmarks[thumb_tip].x < landmarks[thumb_tip - 2].x - 0.03
    else:
        thumb_open_side = landmarks[thumb_tip].x > landmarks[thumb_tip - 2].x + 0.03

    # Deteksi ibu jari mengarah ke atas (vertikal)
    thumb_up = landmarks[thumb_tip].y < landmarks[thumb_mcp].y - 0.05

    # ====== Logika gesture ======
    # ONE: hanya telunjuk terbuka
    if fingers == [True, False, False, False] and not thumb_open_side:
        return "ONE"

    # TWO: telunjuk dan jari tengah terbuka
    elif fingers == [True, True, False, False] and not thumb_open_side:
        return "TWO"

    # FIVE: semua jari dan ibu jari terbuka
    elif all(fingers) and thumb_open_side:
        return "FIVE"

    # FIST: semua jari tertutup
    else:
        closed_fingers = [landmarks[tip].y - landmarks[mcp].y > 0.035 for tip, mcp in zip(finger_tips, finger_mcp)]
        if all(closed_fingers) and not thumb_open_side and not thumb_up:
            return "FIST"

    # THUMB: hanya ibu jari terbuka dan menghadap ke atas
    if thumb_up and all([not f for f in fingers]):
        return "THUMB"

    return None

# ==== Variabel status ====
last_gesture = None
last_time = 0

# ==== Loop utama ====
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False
    result = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    gesture = None
    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_style.get_default_hand_landmarks_style(),
                mp_style.get_default_hand_connections_style()
            )
            gesture = detect_gesture(hand_landmarks, hand_label)

    # ==== Tampilkan teks dan mainkan suara ====
    if gesture and gesture in gesture_messages:
        text = gesture_messages[gesture]
        cv2.putText(frame, text, (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)

        # Delay minimum 2.2 detik antara gesture agar tidak tumpang tindih
        if gesture != last_gesture or (time.time() - last_time) > 2.2:
            last_gesture = gesture
            last_time = time.time()
            play_voice(gesture)

    cv2.imshow("Gesture Recognition (Optimized + THUMB UP)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC untuk keluar
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
