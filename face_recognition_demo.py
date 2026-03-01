import logging as log
import sys
from pathlib import Path
from time import perf_counter
import os 
import re

import cv2
import numpy as np
from openvino import Core, get_version


from landmarks_detector import LandmarksDetector
from face_detector import FaceDetector
from faces_database import FacesDatabase
from face_identifier import FaceIdentifier


log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

class FrameProcessor:
    QUEUE_SIZE = 16

    def __init__(self):
        self.allow_grow = False

        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        core = Core()

        m_fd = os.path.join(CURRENT_DIR,"intel/face-detection-retail-0004/FP16/face-detection-retail-0004.xml")
        self.face_detector = FaceDetector(core, m_fd, (0, 0), confidence_threshold=0.6, roi_scale_factor=1.15)
        m_lm = os.path.join(CURRENT_DIR,"intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml")
        self.landmarks_detector = LandmarksDetector(core, m_lm)
        m_reid = os.path.join(CURRENT_DIR,"intel/face-reidentification-retail-0095/FP16/face-reidentification-retail-0095.xml")
        self.face_identifier = FaceIdentifier(core, m_reid, match_threshold=0.3, match_algo='HUNGARIAN')

        d_fd = "CPU"
        d_lm = "CPU"
        d_reid = "CPU"
        self.face_detector.deploy(d_fd)
        self.landmarks_detector.deploy(d_lm, self.QUEUE_SIZE)
        self.face_identifier.deploy(d_reid, self.QUEUE_SIZE)

        fg = os.path.join(CURRENT_DIR,"face_gallery")
        log.debug('Building faces database using images from {}'.format(fg))
        self.faces_database = FacesDatabase(fg, self.face_identifier, self.landmarks_detector, None, False)
        self.face_identifier.set_faces_database(self.faces_database)
        log.info('Database is built, registered {} identities'.format(len(self.faces_database)))

    def process(self, frame):
        rois = self.face_detector.infer((frame,))
        landmarks = self.landmarks_detector.infer((frame, rois))
        face_identities, _ = self.face_identifier.infer((frame, rois, landmarks))
        return [rois, landmarks, face_identities]


def draw_detections(frame, frame_processor, detections):
    face_frame = {}
    for roi, landmarks, identity in zip(*detections):
        text = frame_processor.face_identifier.get_identity_label(identity.id)
        name = text
        if identity.id != FaceIdentifier.UNKNOWN_ID:
            text += ' %.2f%%' % (100.0 * (1 - identity.distance))

        xmin = max(int(roi.position[0]), 0)
        ymin = max(int(roi.position[1]), 0)
        xmax = min(int(roi.position[0] + roi.size[0]), frame.shape[1])
        ymax = min(int(roi.position[1] + roi.size[1]), frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 220, 0), 2)

        for point in landmarks:
            x = xmin + int(roi.size[0] * point[0])
            y = ymin + int(roi.size[1] * point[1])
            cv2.circle(frame, (x, y), 1, (0, 255, 255), 2)

        try:
            text_cleaned = re.sub(r'_[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}', '', text)
            text_name, _ = text.rsplit("_", 1)
            textsize = cv2.getTextSize(text_cleaned, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            cv2.rectangle(frame, (xmin, ymin - textsize[1]), (xmin + textsize[0], ymin), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, text_cleaned, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        except:
            textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            cv2.rectangle(frame, (xmin, ymin - textsize[1]), (xmin + textsize[0], ymin), (255, 255, 255), cv2.FILLED)
            cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        face_frame[name] = [xmin,ymin,xmax,ymax]
    return frame, face_frame


def main():
    cap = cv2.VideoCapture(2)
    frame_processor = FrameProcessor()
    frame_num = 0
    
    while True:
        start_time = perf_counter()
        ret, frame = cap.read()
        if not ret:
            log.error("Failed to capture frame from camera.")
            break

        detections = frame_processor.process(frame)
        frame, face_frame = draw_detections(frame, frame_processor, detections)  # ✅ FIX

        frame_num += 1
        
        cv2.imshow('Face recognition demo', frame)
        key = cv2.waitKey(1)
        if key in {ord('q'), ord('Q'), 27}:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)