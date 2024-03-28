import cv2
import numpy as np
import tensorflow as tf
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_model = tf.keras.models.load_model('emotion_detection_model_100epochs.h5')
input_video_path = 'a1.mp4'
output_video_path = 'output_video.avi'
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        roi = np.expand_dims(roi_gray, axis=-1)
        roi = np.expand_dims(roi, axis=0)
        roi = roi.astype('float') / 255.0  # Scale

        preds = emotion_model.predict(roi)[0]
        label = class_labels[np.argmax(preds)]
        label_position = (x, y)
        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()
