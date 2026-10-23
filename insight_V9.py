# Install this dependencies in command prompt
"""pip install --upgrade pip
pip install scikit-image lap filterpy
pip install opencv-python opencv-contrib-python
pip install insightface Flask numpy mysql-connector-python deep_sort_realtime"""




from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import os
from datetime import datetime
import insightface
from insightface.app import FaceAnalysis
import mysql.connector
from zoneinfo import ZoneInfo
from deep_sort_realtime.deepsort_tracker import DeepSort
tracker = DeepSort(max_age=30) #Initialize the tracker

app = Flask(__name__)

# Load face analysis model
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_app.prepare(ctx_id=0)  # Use GPU

# Load known faces
known_embeddings = []
known_names = []
KNOWN_FACES_DIR = '/home/ubuntu/fr_simple/known_faces'

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        img = cv2.imread(image_path)
        faces = face_app.get(img)
        if faces:
            known_embeddings.append(faces[0].embedding)
            name = os.path.splitext(filename)[0]
            known_names.append(name)
            print(f"[INFO] Loaded: {name}")
        else:
            print(f"[WARNING] No face found in {filename}")

# MySQL connection (adjust credentials)
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="password",
    database="fr"
)
cursor = db.cursor()
insert_query = "INSERT INTO attendance_insight (name, type, timestamp) VALUES (%s, %s, %s)"

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_face(face_embedding, threshold=0.5):
    if not known_embeddings:
        return "Unknown", None
    
    similarities = np.array([cosine_similarity(face_embedding, emb) for emb in known_embeddings])
    best_idx = np.argmax(similarities)
    
    if similarities[best_idx] > threshold:
        return known_names[best_idx], similarities[best_idx]
    
    return "Unknown", None

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


# 
# Stream frames
last_seen = {}
def gen_frames():
    rtsp_url="/home/ubuntu/office_videos/office_leave_1.mov"
    cap = cv2.VideoCapture(rtsp_url)
    frame_count = 0
    while True:
        frame_count += 1
        if frame_count % 2 == 0:
            if frame_count > 100000:
                frame_count =0
            continue

        success, frame = cap.read()
        if not success:
            continue

        start_x = 650
        start_y = 350
        end_x=1200
        end_y=650


        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 0), 2)

        

        # Crop the central part of the frame
        center_frame = frame[start_y:end_y, start_x:end_x]      
        #center_frame = frame[start_x:end_x,start_y:end_y]  
        faces = face_app.get(center_frame)
        
        now_utc = datetime.now(ZoneInfo("UTC"))
        now_est = now_utc.astimezone(ZoneInfo("America/New_York"))
        
        # --- Step 1: Build detections ---
        detection_info = []
        detections = []
        for face in faces:
            
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            confidence = 1.0  # InsightFace returns 1 face, high confidence
            detections.append(([x1, y1, w, h], 1.0 , face.embedding))
            detection_info.append((box, face.embedding))
            # ---  Step 2: Get tracks from Deep SORT ---
        tracks = tracker.update_tracks(detections, frame=center_frame)   
        # --- Step 3: Process tracks ---
        for track in tracks:
            if not track.is_confirmed():
                continue
        
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            track_box = (l, t, r, b)


             # Validate bounding box
            h_frame, w_frame = center_frame.shape[:2]
            if t < 0 or b > h_frame or l < 0 or r > w_frame or t >= b or l >= r:
                print(f"[WARNING] Skipping invalid crop for Track {track_id}")
                continue


            face_crop = center_frame[t:b, l:r]
            timestamp_str = now_est.strftime('%Y%m%d_%H%M%S')
            
            
            best_iou = 0
            matched_embedding = None
            for det_box, embedding in detection_info:
                iou_score = iou(track_box, tuple(det_box))
                if iou_score > best_iou:
                    best_iou = iou_score
                    matched_embedding = embedding

            if matched_embedding is not None:
                name, similarity = match_face(matched_embedding)
            else:
                name, similarity = "Unknown", None

            # Cooldown logic
            if track_id not in last_seen or (now_est - last_seen[track_id]).total_seconds() > 60:
                last_seen[track_id] = now_est

                if name != "Unknown":
                    image_path = f"detected_faces/{name}_{track_id}_{timestamp_str}.jpg"
                    cv2.imwrite(image_path, face_crop)
                    cv2.rectangle(center_frame, (l, t), (r, b), (0, 255, 0), 2)
                    cv2.putText(center_frame, f"{name} ({similarity:.2f})", (l, t - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    image_path = f"detected_faces/Unknown_{track_id}_{timestamp_str}.jpg"
                    cv2.imwrite(image_path, face_crop)
                    cv2.rectangle(center_frame, (l, t), (r, b), (0, 0, 255), 2)
                    cv2.putText(center_frame, "Unknown", (l, t - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Insert to DB
                cursor.execute(
                    "INSERT INTO attendance_insight_tracking (name, track_id, type, timestamp) VALUES (%s, %s, %s, %s)",
                    (name, track_id, "exit", now_est.strftime('%Y-%m-%d %H:%M:%S'))
                )
                db.commit()


        # Encode and yield frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template_string("""
        <h2>🎥 Live InsightFace Stream</h2>
        <img src="{{ url_for('video_feed') }}">
    """)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    os.makedirs("detected_faces", exist_ok=True)
    app.run(host='0.0.0.0', port=5003)
