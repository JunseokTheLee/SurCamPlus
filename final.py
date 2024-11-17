import cv2
import torch
from facenet_pytorch import MTCNN
import time
import torchvision
import torchvision.transforms as T
import numpy as np
from scipy.spatial import distance
from ultralytics import YOLO
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else'cpu', thresholds=[0.6, 0.7, 0.7], min_face_size=20, margin=20)
yolo_model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

object_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
object_detection_model.eval()
object_detection_model = object_detection_model.to('cuda' if torch.cuda.is_available() else 'cpu')
cap.set(cv2.CAP_PROP_FPS, 25)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()
transform = T.Compose([T.ToTensor()])
prev_positions = {}
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed capture")
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        starttime = time.time()
        boxes, probs = mtcnn.detect(rgb_frame, landmarks=False)
        end_time = time.time()
        
        
        
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob is not None and prob > 0.2:  
                    x1, y1, x2, y2 = [int(coord) for coord in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    
                    
        results = yolo_model(frame)
        detections = results[0]
        count = 0
        current_positions = {}
        
        for idx, detection in enumerate(detections.boxes[:4]):  
            box = detection.xyxy[0].cpu().numpy()
            conf = detection.conf[0].item()
            cls = detection.cls[0].item()
            x1, y1, x2, y2 = map(int, box)
            label = f"{yolo_model.names[int(cls)]}:{conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            current_positions[idx] = (center_x, center_y)
            
            count += 1
        for curr_idx, (curr_x, curr_y) in current_positions.items():
            closest_prev_idx = None
            min_dist = float('inf')
            for prev_idx, (prev_x, prev_y) in prev_positions.items():
                dist = distance.euclidean((curr_x, curr_y), (prev_x, prev_y))
                if dist < min_dist:
                    min_dist = dist
                    closest_prev_idx = prev_idx
            
            
            if closest_prev_idx is not None and min_dist < 50 and min_dist > 5:
                prev_x, prev_y = prev_positions[closest_prev_idx]
                cv2.arrowedLine(frame, (prev_x, prev_y), (int(prev_x + 1.5 * (curr_x - prev_x)), int(prev_y + 1.5 * (curr_y - prev_y))), (0, 255, 255), 4, tipLength=0.3)
        prev_positions = current_positions
        fps = 1/(end_time - starttime)
        cv2.imshow('LIVE DETECTION', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    
    cap.release()
    cv2.destroyAllWindows()
