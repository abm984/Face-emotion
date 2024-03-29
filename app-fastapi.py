from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import tempfile
import os
import tensorflow as tf
import uvicorn

app = FastAPI()
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_model = tf.keras.models.load_model('emotion_detection_model_100epochs.h5')

@app.post("/process_video/")
async def process_video(video_file: UploadFile = File(...)):
    try:
        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = os.path.join(temp_dir, video_file.filename)
            with open(video_path, "wb") as buffer:
                buffer.write(await video_file.read())

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise HTTPException(status_code=500, detail="Error: Could not open video file.")

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_video_path = os.path.join(temp_dir, "output_video.avi")
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
                    roi = roi.astype('float') / 255.0  

                    preds = emotion_model.predict(roi)[0]
                    label = class_labels[np.argmax(preds)]
                    label_position = (x, y)
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(frame)
            cap.release()
            out.release()
            return StreamingResponse(open(output_video_path, "rb"), media_type="video/x-msvideo")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
import uvicorn
if __name__ == "__main__":
   uvicorn.run(app, host="127.0.0.1", port=8000)