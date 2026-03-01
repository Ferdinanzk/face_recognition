import logging as log
import sys
import os 
import re
import math
import cv2
import numpy as np
from time import perf_counter
from openvino import Core, get_version

# Verify imports exist
try:
    from landmarks_detector import LandmarksDetector
    from face_detector import FaceDetector
    from faces_database import FacesDatabase
    from face_identifier import FaceIdentifier
except ImportError as e:
    print(f"\n[ERROR] Missing custom file: {e}")
    print("Ensure 'landmarks_detector.py', 'face_detector.py', etc., are in the same folder.\n")
    sys.exit(1)

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# ---------------------------------------------------------
# POSE DETECTOR (Robust Version)
# ---------------------------------------------------------
class PoseDetector:
    def __init__(self, core, model_path, device="CPU"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pose model XML not found at: {model_path}")

        log.info(f"Loading Pose Model: {model_path}")
        self.model = core.read_model(model_path)
        self.compiled_model = core.compile_model(self.model, device)
        
        # Robust output layer finding
        self.output_layer_heatmaps = None
        for output in self.compiled_model.outputs:
            shape = output.shape
            # Heatmaps usually have 19 or 18 channels (Keypoints)
            if shape[1] == 19 or shape[1] == 18: 
                self.output_layer_heatmaps = output
                break
        
        if self.output_layer_heatmaps is None:
            # Fallback to index 1 if auto-detection fails
            self.output_layer_heatmaps = self.compiled_model.outputs[1]

        self.input_layer = self.compiled_model.inputs[0]
        self.input_shape = self.input_layer.shape
        self.h, self.w = self.input_shape[2], self.input_shape[3]
        
        self.POSE_PAIRS = [
            (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), 
            (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), 
            (0, 1), (0, 14), (0, 15), (14, 16), (15, 17)
        ]

    def infer(self, frame):
        try:
            resized_frame = cv2.resize(frame, (self.w, self.h))
            input_data = np.expand_dims(np.transpose(resized_frame, (2, 0, 1)), 0)
            results = self.compiled_model([input_data])
            heatmaps = results[self.output_layer_heatmaps][0]
            return self.parse_heatmaps(heatmaps, frame.shape)
        except Exception as e:
            log.error(f"Pose Inference Failed: {e}")
            return []

    def parse_heatmaps(self, heatmaps, original_shape):
        orig_h, orig_w = original_shape[:2]
        skeletons = []
        keypoints = {} 
        conf_thresh = 0.5
        
        for i in range(len(heatmaps) - 1): 
            heatmap = heatmaps[i]
            _, conf, _, point = cv2.minMaxLoc(heatmap)
            if conf > conf_thresh:
                x = int(point[0] * orig_w / self.w)
                y = int(point[1] * orig_h / self.h)
                keypoints[i] = (x, y)
        
        if 0 in keypoints or 1 in keypoints:
            skeletons.append({'keypoints': keypoints})
        return skeletons

# ---------------------------------------------------------
# PERSON TRACKER (Crash-Proof Version)
# ---------------------------------------------------------
class PersonTracker:
    def __init__(self, max_disappeared=30):
        self.next_object_id = 0
        self.objects = {} 
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def update(self, detected_skeletons):
        if not detected_skeletons:
            # If no skeletons detected, mark existing as disappeared
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = []
        for skel in detected_skeletons:
            pts = skel['keypoints']
            # Prefer Neck(1) -> Nose(0) -> First available
            if 1 in pts: c = pts[1]
            elif 0 in pts: c = pts[0]
            elif len(pts) > 0: c = list(pts.values())[0]
            else: c = (0,0)
            input_centroids.append(c)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], detected_skeletons[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = [self.objects[oid]['centroid'] for oid in object_ids]

            # Calculate Distance Matrix Manually
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    dx = object_centroids[i][0] - input_centroids[j][0]
                    dy = object_centroids[i][1] - input_centroids[j][1]
                    D[i, j] = math.sqrt(dx*dx + dy*dy)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols: continue
                if D[row, col] > 200: continue # Distance threshold

                object_id = object_ids[row]
                self.objects[object_id]['centroid'] = input_centroids[col]
                self.objects[object_id]['skeleton'] = detected_skeletons[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], detected_skeletons[col])

        return self.objects

    def register(self, centroid, skeleton):
        self.objects[self.next_object_id] = {'centroid': centroid, 'name': 'Scanning...', 'skeleton': skeleton}
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update_name(self, object_id, name):
        if object_id in self.objects:
            self.objects[object_id]['name'] = name

# ---------------------------------------------------------
# MAIN PROCESSOR
# ---------------------------------------------------------
class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self):
        log.info('Initializing OpenVINO...')
        core = Core()

        # Define Paths
        m_fd = os.path.join(CURRENT_DIR, "intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml")
        m_lm = os.path.join(CURRENT_DIR, "intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml")
        m_reid = os.path.join(CURRENT_DIR, "intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml")
        m_pose = os.path.join(CURRENT_DIR, "intel\human-pose-estimation-0001")

        # Load Face Models
        try:
            self.face_detector = FaceDetector(core, m_fd, (0, 0), confidence_threshold=0.6, roi_scale_factor=1.15)
            self.landmarks_detector = LandmarksDetector(core, m_lm)
            self.face_identifier = FaceIdentifier(core, m_reid, match_threshold=0.3, match_algo='HUNGARIAN')
            self.face_detector.deploy("CPU")
            self.landmarks_detector.deploy("CPU", self.QUEUE_SIZE)
            self.face_identifier.deploy("CPU", self.QUEUE_SIZE)
        except Exception as e:
            log.error(f"Error loading Face Models: {e}")
            sys.exit(1)

        # Load Pose Model
        try:
            self.pose_detector = PoseDetector(core, m_pose, "CPU")
            log.info("Pose detection model loaded successfully.")
        except Exception as e:
            log.warning(f"SKIPPING POSE DETECTION: {e}")
            self.pose_detector = None

        # Database
        fg = os.path.join(CURRENT_DIR, "face_gallery")
        if not os.path.exists(fg):
            os.makedirs(fg)
            log.warning(f"Created empty face_gallery folder at {fg}")

        self.faces_database = FacesDatabase(fg, self.face_identifier, self.landmarks_detector, None, False)
        self.face_identifier.set_faces_database(self.faces_database)
        
        self.tracker = PersonTracker()

    def process(self, frame):
        rois = self.face_detector.infer((frame,))
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, _ = self.face_identifier.infer((frame, rois, landmarks))
        
        skeletons = []
        if self.pose_detector:
            skeletons = self.pose_detector.infer(frame)
            
        tracked_objects = self.tracker.update(skeletons)
        
        # Match Faces to Skeletons
        for obj_id, obj_data in tracked_objects.items():
            pts = obj_data['skeleton']['keypoints']
            head_pt = pts.get(0, pts.get(1, None))
            
            if head_pt:
                for i, roi in enumerate(rois):
                    xmin, ymin = roi.position
                    xmax, ymax = xmin + roi.size[0], ymin + roi.size[1]
                    if xmin < head_pt[0] < xmax and ymin < head_pt[1] < ymax:
                        identity = face_identities[i]
                        text = self.face_identifier.get_identity_label(identity.id)
                        text_cleaned = re.sub(r'_[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}', '', text)
                        if identity.id != FaceIdentifier.UNKNOWN_ID:
                             self.tracker.update_name(obj_id, text_cleaned)

        return [rois, landmarks, face_identities, tracked_objects]

def draw_detections(frame, frame_processor, detections):
    rois, landmarks, identities, tracked_objects = detections
    
    # Draw Skeletons
    if frame_processor.pose_detector:
        for obj_id, data in tracked_objects.items():
            pts = data['skeleton']['keypoints']
            name = data['name']
            
            for pair in frame_processor.pose_detector.POSE_PAIRS:
                partA, partB = pair
                if partA in pts and partB in pts:
                    cv2.line(frame, pts[partA], pts[partB], (0, 255, 255), 2)
            
            head_pt = pts.get(0, pts.get(1, list(pts.values())[0] if pts else None))
            if head_pt:
                cv2.putText(frame, f"{name}", (head_pt[0], head_pt[1] - 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Draw Faces
    for roi, identity in zip(rois, identities):
        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), frame.shape[1])
        ymax = min(int(roi.position[1] + roi.size[1]), frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)
        
        # Show name on box as well
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        text_cleaned = re.sub(r'_[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}', '', text)
        cv2.putText(frame, text_cleaned, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return frame

def main():
    print("Starting Camera...")
    cap = cv2.VideoCapture(0) # Changed to 0 (default cam), change to 2 if using external
    if not cap.isOpened():
        log.error("Cannot open webcam.")
        return

    try:
        frame_processor = FrameProcessor()
    except Exception as e:
        log.critical(f"Failed to initialize Processor: {e}")
        return

    while True:
        ret, frame = cap.read()
        if not ret: break

        detections = frame_processor.process(