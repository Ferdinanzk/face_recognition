"""
skeleton_and_facerecognition.py

Combined MediaPipe skeleton tracking + OpenVINO face recognition.
Supports multiple people simultaneously.

Architecture:
  1. OpenVINO face detector runs every frame to locate all faces.
  2. Each detected face is matched to a TrackedPerson by centre-distance.
  3. For each person, a body crop is estimated from their face bbox and
     MediaPipe Pose runs on that crop (static_image_mode — no cross-person
     temporal confusion).
  4. Face recognition (landmarks + re-identification) runs periodically
     on detected face ROIs.  Each person has their own majority-vote buffer.
  5. When a person disappears for PERSON_TIMEOUT frames they are removed;
     re-entering triggers a fresh identification.
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp

# ── PATH SETUP ────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from face_detector import FaceDetector          # noqa: E402
from face_identifier import FaceIdentifier      # noqa: E402
from landmarks_detector import LandmarksDetector  # noqa: E402
from faces_database import FacesDatabase        # noqa: E402
from openvino import Core                       # noqa: E402

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
INTEL_MODELS = os.path.join(HERE, "intel")
FACE_GALLERY = os.path.join(HERE, "face_gallery")
DEVICE = "CPU"

FACE_DET_MODEL = os.path.join(
    INTEL_MODELS, "face-detection-adas-0001", "FP32", "face-detection-adas-0001.xml"
)
FACE_REID_MODEL = os.path.join(
    INTEL_MODELS, "face-reidentification-retail-0095", "FP32",
    "face-reidentification-retail-0095.xml"
)
LANDMARKS_MODEL = os.path.join(
    INTEL_MODELS, "landmarks-regression-retail-0009", "FP32",
    "landmarks-regression-retail-0009.xml"
)

FACE_DET_CONFIDENCE    = 0.6
FACE_MATCH_THRESHOLD   = 0.3   # cosine-distance — lower = stricter (rejects strangers)
 
# Max simultaneous faces the inference queue can handle.
MAX_FACES = 10

# Frames without a detection before a person is dropped.
PERSON_TIMEOUT = 30

# How often (frames) face recognition re-runs to verify / correct identity.
FACE_REC_COOLDOWN_FRAMES = 30

# Shorter cooldown when the result was Unknown — retry faster.
UNKNOWN_COOLDOWN_FRAMES = 10

# Majority-vote window per person.
VOTE_WINDOW = 5

# Camera index (try this first, fall back to 0).
CAMERA_INDEX = 2

# ── MEDIAPIPE ─────────────────────────────────────────────────────────────────
mp_pose = mp.solutions.pose

SKELETON_COLOR = (0, 200, 255)
JOINT_COLOR    = (0, 255, 0)
NAME_COLOR     = (0, 255, 255)


# ── TRACKED PERSON ────────────────────────────────────────────────────────────

class TrackedPerson:
    _next_id = 0

    def __init__(self, face_bbox):
        self.id = TrackedPerson._next_id
        TrackedPerson._next_id += 1
        self.face_bbox = face_bbox          # (x1, y1, x2, y2) in frame coords
        self.name = None
        self.vote_buffer = []
        self.pose_landmarks = None          # MediaPipe NormalizedLandmarkList
        self.crop_offset = (0, 0)           # (x, y) top-left of body crop
        self.crop_size = (0, 0)             # (w, h) of body crop
        self.frames_since_seen = 0

    def update_face(self, face_bbox):
        self.face_bbox = face_bbox
        self.frames_since_seen = 0

    def add_name_vote(self, name, distance):
        self.vote_buffer.append((name, distance))
        if len(self.vote_buffer) > VOTE_WINDOW:
            self.vote_buffer.pop(0)
        if not self.vote_buffer:
            return
        # Count votes per name
        name_counts = {}
        name_best_dist = {}
        for vname, vdist in self.vote_buffer:
            name_counts[vname] = name_counts.get(vname, 0) + 1
            if vname not in name_best_dist or vdist < name_best_dist[vname]:
                name_best_dist[vname] = vdist
        majority = max(name_counts, key=name_counts.get)
        count = name_counts[majority]
        # Require at least 2 votes for the same name to commit
        if count >= 2 and majority != self.name:
            self.name = majority
            print(f"[INFO] Person {self.id} → {self.name} "
                  f"(votes {count}/{len(self.vote_buffer)}, "
                  f"best dist {name_best_dist[majority]:.3f})")


# ── GEOMETRY HELPERS ──────────────────────────────────────────────────────────

def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def center_dist(b1, b2):
    c1 = bbox_center(b1)
    c2 = bbox_center(b2)
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def estimate_body_region(face_bbox, frame_h, frame_w):
    """Estimate full-body crop from a face bounding box."""
    x1, y1, x2, y2 = face_bbox
    face_w = x2 - x1
    face_h = y2 - y1
    cx = (x1 + x2) // 2

    body_w = int(face_w * 4)
    body_h = int(face_h * 8)

    bx1 = max(0, cx - body_w // 2)
    by1 = max(0, y1 - int(face_h * 0.3))
    bx2 = min(frame_w, cx + body_w // 2)
    by2 = min(frame_h, by1 + body_h)

    return bx1, by1, bx2, by2


def roi_to_bbox(roi):
    x1 = int(roi.position[0])
    y1 = int(roi.position[1])
    x2 = int(roi.position[0] + roi.size[0])
    y2 = int(roi.position[1] + roi.size[1])
    return (x1, y1, x2, y2)


# ── MATCHING ──────────────────────────────────────────────────────────────────

def match_detections_to_persons(face_bboxes, tracked, max_dist=150):
    """
    Greedy nearest-centre matching.
    Returns (matches, unmatched_det_indices, unmatched_person_ids).
    """
    if not tracked or not face_bboxes:
        return [], list(range(len(face_bboxes))), list(tracked.keys())

    person_ids = list(tracked.keys())
    pairs = []
    for d_idx, det_bbox in enumerate(face_bboxes):
        for p_id in person_ids:
            d = center_dist(det_bbox, tracked[p_id].face_bbox)
            pairs.append((d, d_idx, p_id))
    pairs.sort()

    used_dets = set()
    used_pids = set()
    matches = []
    for d, d_idx, p_id in pairs:
        if d_idx in used_dets or p_id in used_pids:
            continue
        if d > max_dist:
            break
        matches.append((d_idx, p_id))
        used_dets.add(d_idx)
        used_pids.add(p_id)

    unmatched_dets = [i for i in range(len(face_bboxes)) if i not in used_dets]
    unmatched_pids = [p for p in person_ids if p not in used_pids]
    return matches, unmatched_dets, unmatched_pids


# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def build_face_recognition_system():
    core = Core()

    face_det = FaceDetector(
        core, FACE_DET_MODEL, input_size=(0, 0),
        confidence_threshold=FACE_DET_CONFIDENCE
    )
    face_det.deploy(DEVICE)

    lm_det = LandmarksDetector(core, LANDMARKS_MODEL)
    lm_det.deploy(DEVICE, max_requests=MAX_FACES)

    face_id = FaceIdentifier(
        core, FACE_REID_MODEL,
        match_threshold=FACE_MATCH_THRESHOLD,
        match_algo="MIN_DIST"
    )
    face_id.deploy(DEVICE, max_requests=MAX_FACES)

    faces_db = FacesDatabase(FACE_GALLERY, face_id, lm_det)
    face_id.set_faces_database(faces_db)

    print(f"[INFO] Gallery loaded — {len(faces_db)} "
          f"identit{'y' if len(faces_db) == 1 else 'ies'}.")
    return face_det, lm_det, face_id


# ── FACE RECOGNITION ─────────────────────────────────────────────────────────

def recognize_faces(frame, rois, lm_det, face_id):
    """
    Run landmarks + re-identification on pre-detected face ROIs.
    Returns list of (name, distance, bbox).
    """
    if not rois:
        return []

    landmarks = lm_det.infer((frame, rois))

    # Clear stale outputs before enqueueing new requests
    face_id.clear()
    face_id.start_async(frame, rois, landmarks)
    results, _ = face_id.postprocess()

    recognised = []
    for roi, result in zip(rois, results):
        name = face_id.get_identity_label(result.id)
        recognised.append((name, result.distance, roi_to_bbox(roi)))
    return recognised


# ── DRAWING ───────────────────────────────────────────────────────────────────

def draw_person(frame, person):
    """Draw skeleton + name for one tracked person."""
    lm = person.pose_landmarks
    ox, oy = person.crop_offset
    cw, ch = person.crop_size

    if lm is not None:
        # Connections
        for c0, c1 in mp_pose.POSE_CONNECTIONS:
            s = lm.landmark[c0]
            e = lm.landmark[c1]
            if s.visibility < 0.3 or e.visibility < 0.3:
                continue
            sp = (int(s.x * cw + ox), int(s.y * ch + oy))
            ep = (int(e.x * cw + ox), int(e.y * ch + oy))
            cv2.line(frame, sp, ep, SKELETON_COLOR, 2)

        # Joints
        for mark in lm.landmark:
            if mark.visibility < 0.3:
                continue
            pt = (int(mark.x * cw + ox), int(mark.y * ch + oy))
            cv2.circle(frame, pt, 3, JOINT_COLOR, -1)

    # ── Name label ────────────────────────────────────────────
    if not person.name:
        return

    # Anchor to nose if skeleton available, otherwise above face bbox
    head_x, head_y = None, None
    if lm is not None:
        nose = lm.landmark[mp_pose.PoseLandmark.NOSE]
        if nose.visibility >= 0.3:
            head_x = int(nose.x * cw + ox)
            head_y = int(nose.y * ch + oy)

    if head_x is None:
        x1, y1, x2, y2 = person.face_bbox
        head_x = (x1 + x2) // 2
        head_y = y1

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.85, 2
    (tw, th), baseline = cv2.getTextSize(person.name, font, scale, thick)
    tx = head_x - tw // 2
    ty = head_y - 35
    pad = 5

    cv2.rectangle(frame,
                  (tx - pad, ty - th - pad),
                  (tx + tw + pad, ty + baseline + pad),
                  (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, person.name, (tx, ty), font, scale, NAME_COLOR, thick)
    cv2.line(frame, (head_x, ty + baseline), (head_x, head_y), NAME_COLOR, 1)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("[INFO] Loading models …")
    face_det, lm_det, face_id = build_face_recognition_system()
    print("[INFO] Models ready.")

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[WARN] Camera {CAMERA_INDEX} unavailable — trying 0.")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] No camera found.")
        return

    # static_image_mode=True so each crop is treated independently —
    # no cross-person temporal tracking confusion.
    pose = mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.3,
        model_complexity=1
    )

    tracked = {}               # person_id → TrackedPerson
    face_rec_cooldown = 0

    print("[INFO] Press Q to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]

        # ── 1. Face detection (every frame — fast on OpenVINO) ────
        face_rois = face_det.infer((frame,))
        face_bboxes = [roi_to_bbox(roi) for roi in face_rois]

        # ── 2. Match detections → tracked persons ────────────────
        matches, new_dets, lost_pids = match_detections_to_persons(
            face_bboxes, tracked
        )

        for det_idx, pid in matches:
            tracked[pid].update_face(face_bboxes[det_idx])

        for det_idx in new_dets:
            p = TrackedPerson(face_bboxes[det_idx])
            tracked[p.id] = p

        for pid in lost_pids:
            tracked[pid].frames_since_seen += 1

        # Drop persons gone too long
        gone = [pid for pid, p in tracked.items()
                if p.frames_since_seen > PERSON_TIMEOUT]
        for pid in gone:
            print(f"[INFO] Person {pid} ({tracked[pid].name or '?'}) timed out.")
            del tracked[pid]

        # ── 3. Skeleton per person (body crop → MediaPipe Pose) ──
        for person in tracked.values():
            if person.frames_since_seen > 0:
                person.pose_landmarks = None
                continue

            bx1, by1, bx2, by2 = estimate_body_region(
                person.face_bbox, fh, fw
            )
            crop_w = bx2 - bx1
            crop_h = by2 - by1

            if crop_w < 50 or crop_h < 50:
                person.pose_landmarks = None
                continue

            crop = frame[by1:by2, bx1:bx2]
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_crop)

            person.pose_landmarks = result.pose_landmarks
            person.crop_offset = (bx1, by1)
            person.crop_size = (crop_w, crop_h)

        # ── 4. Face recognition (periodic) ───────────────────────
        if face_rec_cooldown > 0:
            face_rec_cooldown -= 1

        if face_rec_cooldown <= 0 and face_rois:
            recognised = recognize_faces(frame, face_rois, lm_det, face_id)

            has_known = False
            for rec_name, rec_dist, rec_bbox in recognised:
                if rec_name == FaceIdentifier.UNKNOWN_ID_LABEL:
                    continue
                has_known = True

                # Associate with closest tracked person
                best_person = None
                best_d = float("inf")
                for p in tracked.values():
                    d = center_dist(rec_bbox, p.face_bbox)
                    if d < best_d:
                        best_d = d
                        best_person = p

                if best_person is not None and best_d < 150:
                    best_person.add_name_vote(rec_name, rec_dist)

            face_rec_cooldown = (FACE_REC_COOLDOWN_FRAMES
                                 if has_known else UNKNOWN_COOLDOWN_FRAMES)

        # ── 5. Draw (skeleton only for recognised gallery members) ─
        for person in tracked.values():
            if person.frames_since_seen == 0 and person.name is not None:
                draw_person(frame, person)

        # HUD
        cv2.putText(frame, f"Persons: {len(tracked)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Skeleton + Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
