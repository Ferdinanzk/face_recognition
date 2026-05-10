"""
Skeleton detector using Intel's human-pose-estimation-0001 OpenVINO model (OpenPose-based).

Model must be downloaded first:
  intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.xml
  intel/human-pose-estimation-0001/FP16/human-pose-estimation-0001.bin

Download from:
  https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/human-pose-estimation-0001/FP16/
"""

import logging as log
import numpy as np
import cv2

try:
    from numpy.core.umath import clip as np_clip
except ImportError:
    from numpy import clip as np_clip


# ---------------------------------------------------------------------------
# OpenPose decoder (adapted from open_model_zoo open_pose.py)
# ---------------------------------------------------------------------------

class OpenPoseDecoder:
    BODY_PARTS_KPT_IDS = (
        (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
        (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17),
        (2, 16), (5, 17)
    )
    BODY_PARTS_PAF_IDS = (12, 20, 14, 16, 22, 24, 0, 2, 4, 6, 8, 10, 28, 30, 34, 32, 36, 18, 26)

    def __init__(self, num_joints=18, skeleton=BODY_PARTS_KPT_IDS,
                 paf_indices=BODY_PARTS_PAF_IDS, max_points=100,
                 score_threshold=0.1, min_paf_alignment_score=0.05, delta=0.5):
        self.num_joints = num_joints
        self.skeleton = skeleton
        self.paf_indices = paf_indices
        self.max_points = max_points
        self.score_threshold = score_threshold
        self.min_paf_alignment_score = min_paf_alignment_score
        self.delta = delta
        self.points_per_limb = 10
        self.grid = np.arange(self.points_per_limb, dtype=np.float32).reshape(1, -1, 1)

    def __call__(self, heatmaps, nms_heatmaps, pafs):
        batch_size, _, h, w = heatmaps.shape
        assert batch_size == 1, 'Batch size of 1 only supported'
        keypoints = self._extract_points(heatmaps, nms_heatmaps)
        pafs = np.transpose(pafs, (0, 2, 3, 1))
        if self.delta > 0:
            for kpts in keypoints:
                kpts[:, :2] += self.delta
                np_clip(kpts[:, 0], 0, w - 1, out=kpts[:, 0])
                np_clip(kpts[:, 1], 0, h - 1, out=kpts[:, 1])
        pose_entries, all_keypoints = self._group_keypoints(
            keypoints, pafs, pose_entry_size=self.num_joints + 2)
        poses, scores = self._convert_to_coco_format(pose_entries, all_keypoints)
        if len(poses) > 0:
            poses = np.asarray(poses, dtype=np.float32).reshape((len(poses), -1, 3))
        else:
            poses = np.empty((0, 17, 3), dtype=np.float32)
            scores = np.empty(0, dtype=np.float32)
        return poses, scores

    def _extract_points(self, heatmaps, nms_heatmaps):
        _, _, h, w = heatmaps.shape
        xs, ys, scores = self._top_k(nms_heatmaps)
        masks = scores > self.score_threshold
        all_keypoints = []
        keypoint_id = 0
        for k in range(self.num_joints):
            mask = masks[0, k]
            x = xs[0, k][mask].ravel()
            y = ys[0, k][mask].ravel()
            score = scores[0, k][mask].ravel()
            n = len(x)
            if n == 0:
                all_keypoints.append(np.empty((0, 4), dtype=np.float32))
                continue
            x, y = self._refine(heatmaps[0, k], x, y)
            np_clip(x, 0, w - 1, out=x)
            np_clip(y, 0, h - 1, out=y)
            kpts = np.empty((n, 4), dtype=np.float32)
            kpts[:, 0] = x
            kpts[:, 1] = y
            kpts[:, 2] = score
            kpts[:, 3] = np.arange(keypoint_id, keypoint_id + n)
            keypoint_id += n
            all_keypoints.append(kpts)
        return all_keypoints

    def _top_k(self, heatmaps):
        N, K, _, W = heatmaps.shape
        heatmaps = heatmaps.reshape(N, K, -1)
        ind = heatmaps.argpartition(-self.max_points, axis=2)[:, :, -self.max_points:]
        scores = np.take_along_axis(heatmaps, ind, axis=2)
        subind = np.argsort(-scores, axis=2)
        ind = np.take_along_axis(ind, subind, axis=2)
        scores = np.take_along_axis(scores, subind, axis=2)
        y, x = np.divmod(ind, W)
        return x, y, scores

    @staticmethod
    def _refine(heatmap, x, y):
        h, w = heatmap.shape[-2:]
        valid = np.logical_and(
            np.logical_and(x > 0, x < w - 1),
            np.logical_and(y > 0, y < h - 1))
        xx, yy = x[valid], y[valid]
        dx = np.sign(heatmap[yy, xx + 1] - heatmap[yy, xx - 1], dtype=np.float32) * 0.25
        dy = np.sign(heatmap[yy + 1, xx] - heatmap[yy - 1, xx], dtype=np.float32) * 0.25
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        x[valid] += dx
        y[valid] += dy
        return x, y

    @staticmethod
    def _is_disjoint(pose_a, pose_b):
        a, b = pose_a[:-2], pose_b[:-2]
        return np.all(np.logical_or.reduce((a == b, a < 0, b < 0)))

    def _update_poses(self, kpt_a_id, kpt_b_id, all_keypoints,
                      connections, pose_entries, pose_entry_size):
        for connection in connections:
            pose_a_idx = -1
            pose_b_idx = -1
            for j, pose in enumerate(pose_entries):
                if pose[kpt_a_id] == connection[0]:
                    pose_a_idx = j
                if pose[kpt_b_id] == connection[1]:
                    pose_b_idx = j
            if pose_a_idx < 0 and pose_b_idx < 0:
                entry = np.full(pose_entry_size, -1, dtype=np.float32)
                entry[kpt_a_id] = connection[0]
                entry[kpt_b_id] = connection[1]
                entry[-1] = 2
                entry[-2] = np.sum(all_keypoints[connection[0:2], 2]) + connection[2]
                pose_entries.append(entry)
            elif pose_a_idx >= 0 and pose_b_idx >= 0 and pose_a_idx != pose_b_idx:
                pa = pose_entries[pose_a_idx]
                pb = pose_entries[pose_b_idx]
                if self._is_disjoint(pa, pb):
                    pa += pb
                    pa[:-2] += 1
                    pa[-2] += connection[2]
                    del pose_entries[pose_b_idx]
            elif pose_a_idx >= 0 and pose_b_idx >= 0:
                pose_entries[pose_a_idx][-2] += connection[2]
            elif pose_a_idx >= 0:
                pose = pose_entries[pose_a_idx]
                if pose[kpt_b_id] < 0:
                    pose[-2] += all_keypoints[connection[1], 2]
                pose[kpt_b_id] = connection[1]
                pose[-2] += connection[2]
                pose[-1] += 1
            elif pose_b_idx >= 0:
                pose = pose_entries[pose_b_idx]
                if pose[kpt_a_id] < 0:
                    pose[-2] += all_keypoints[connection[0], 2]
                pose[kpt_a_id] = connection[0]
                pose[-2] += connection[2]
                pose[-1] += 1
        return pose_entries

    @staticmethod
    def _connections_nms(a_idx, b_idx, affinity_scores):
        order = affinity_scores.argsort()[::-1]
        affinity_scores = affinity_scores[order]
        a_idx = a_idx[order]
        b_idx = b_idx[order]
        idx = []
        has_a = set()
        has_b = set()
        for t, (i, j) in enumerate(zip(a_idx, b_idx)):
            if i not in has_a and j not in has_b:
                idx.append(t)
                has_a.add(i)
                has_b.add(j)
        idx = np.asarray(idx, dtype=np.int32)
        return a_idx[idx], b_idx[idx], affinity_scores[idx]

    def _group_keypoints(self, all_keypoints_by_type, pafs, pose_entry_size=20):
        all_keypoints = np.concatenate(all_keypoints_by_type, axis=0)
        pose_entries = []
        for part_id, paf_channel in enumerate(self.paf_indices):
            kpt_a_id, kpt_b_id = self.skeleton[part_id]
            kpts_a = all_keypoints_by_type[kpt_a_id]
            kpts_b = all_keypoints_by_type[kpt_b_id]
            n, m = len(kpts_a), len(kpts_b)
            if n == 0 or m == 0:
                continue
            a = kpts_a[:, :2]
            a = np.broadcast_to(a[None], (m, n, 2))
            b = kpts_b[:, :2]
            vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)
            steps = (1 / (self.points_per_limb - 1) * vec_raw)
            points = steps * self.grid + a.reshape(-1, 1, 2)
            points = points.round().astype(np.int32)
            x = points[..., 0].ravel()
            y = points[..., 1].ravel()
            part_pafs = pafs[0, :, :, paf_channel:paf_channel + 2]
            field = part_pafs[y, x].reshape(-1, self.points_per_limb, 2)
            vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
            vec = vec_raw / (vec_norm + 1e-6)
            affinity_scores = (field * vec).sum(-1).reshape(-1, self.points_per_limb)
            valid_aff = affinity_scores > self.min_paf_alignment_score
            valid_num = valid_aff.sum(1)
            affinity_scores = (affinity_scores * valid_aff).sum(1) / (valid_num + 1e-6)
            success_ratio = valid_num / self.points_per_limb
            valid_limbs = np.where(
                np.logical_and(affinity_scores > 0, success_ratio > 0.8))[0]
            if len(valid_limbs) == 0:
                continue
            b_idx, a_idx = np.divmod(valid_limbs, n)
            affinity_scores = affinity_scores[valid_limbs]
            a_idx, b_idx, affinity_scores = self._connections_nms(
                a_idx, b_idx, affinity_scores)
            connections = list(zip(
                kpts_a[a_idx, 3].astype(np.int32),
                kpts_b[b_idx, 3].astype(np.int32),
                affinity_scores))
            if not connections:
                continue
            pose_entries = self._update_poses(
                kpt_a_id, kpt_b_id, all_keypoints,
                connections, pose_entries, pose_entry_size)
        pose_entries = np.asarray(pose_entries, dtype=np.float32).reshape(
            -1, pose_entry_size)
        pose_entries = pose_entries[pose_entries[:, -1] >= 3]
        return pose_entries, all_keypoints

    @staticmethod
    def _convert_to_coco_format(pose_entries, all_keypoints):
        num_joints = 17
        coco_keypoints = []
        scores = []
        reorder_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        for pose in pose_entries:
            if len(pose) == 0:
                continue
            keypoints = np.zeros(num_joints * 3)
            person_score = pose[-2]
            for keypoint_id, target_id in zip(pose[:-2], reorder_map):
                if target_id < 0:
                    continue
                cx, cy, score = 0, 0, 0
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                keypoints[target_id * 3 + 0] = cx
                keypoints[target_id * 3 + 1] = cy
                keypoints[target_id * 3 + 2] = score
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose[-1] - 1)))
        return np.asarray(coco_keypoints), np.asarray(scores)


# ---------------------------------------------------------------------------
# SkeletonDetector
# ---------------------------------------------------------------------------

class SkeletonDetector:
    """
    Detects human skeletons using Intel human-pose-estimation-0001 (OpenPose).

    Output poses are arrays of shape (17, 3): 17 COCO keypoints each with (x, y, score).

    COCO keypoint order:
      0:nose  1:left_eye  2:right_eye  3:left_ear  4:right_ear
      5:left_shoulder  6:right_shoulder  7:left_elbow  8:right_elbow
      9:left_wrist  10:right_wrist  11:left_hip  12:right_hip
      13:left_knee  14:right_knee  15:left_ankle  16:right_ankle
    """

    # Skeleton limb connections (COCO 17 keypoints)
    SKELETON = (
        (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
        (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9),
        (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
    )

    # Colors per keypoint (17 colors)
    COLORS = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255)
    )

    def __init__(self, core, model_path, device='CPU', conf_threshold=0.1):
        log.info('Reading Skeleton (pose) model: {}'.format(model_path))
        model = core.read_model(model_path)

        # Identify PAF vs heatmap outputs by channel count (PAF has 2x more channels)
        out0_shape = list(model.outputs[0].shape)
        out1_shape = list(model.outputs[1].shape)
        if out0_shape[1] > out1_shape[1]:
            paf_idx, hm_idx = 0, 1
        else:
            paf_idx, hm_idx = 1, 0

        compiled = core.compile_model(model, device)
        self._request = compiled.create_infer_request()
        self._input_name = model.inputs[0].get_any_name()
        self._paf_out = compiled.outputs[paf_idx]
        self._hm_out = compiled.outputs[hm_idx]

        # Spatial scale: ratio of (model input height) to (heatmap output height)
        input_h = model.inputs[0].shape[2]
        hm_h = compiled.outputs[hm_idx].shape[2]
        self._output_scale = input_h / hm_h
        self._input_h = input_h
        self._input_w = model.inputs[0].shape[3]

        # NMS kernel (mimics the max-pool NMS from open_pose.py)
        p = max(0, int(np.round(6 / 7 * self._output_scale)))
        self._nms_kernel = 2 * p + 1

        num_joints = compiled.outputs[hm_idx].shape[1] - 1  # last channel = background
        self._decoder = OpenPoseDecoder(num_joints, score_threshold=conf_threshold)
        self._conf_threshold = conf_threshold
        log.info('Skeleton model ready. output_scale={:.1f}, nms_kernel={}'.format(
            self._output_scale, self._nms_kernel))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def infer(self, frame):
        """
        Run pose estimation on a frame.
        Returns: list of np.ndarray, each shape (17, 3) — (x, y, score) per keypoint,
                 in original frame pixel coordinates.
        """
        preprocessed, scale_xy = self._preprocess(frame)
        self._request.infer({self._input_name: preprocessed})
        heatmaps = self._request.get_tensor(self._hm_out).data.copy()
        pafs = self._request.get_tensor(self._paf_out).data.copy()
        nms_heatmaps = self._apply_nms(heatmaps)
        poses, _ = self._decoder(heatmaps, nms_heatmaps, pafs)
        if len(poses) > 0:
            # Scale from heatmap coords → original frame coords
            poses[:, :, 0] *= scale_xy[0] * self._output_scale
            poses[:, :, 1] *= scale_xy[1] * self._output_scale
        return list(poses)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_pose_bbox(pose, min_score=0.1):
        """
        Bounding box covering all confident keypoints.
        Returns (x1, y1, x2, y2) or None if no keypoints are confident.
        """
        valid = pose[:, 2] > min_score
        if not np.any(valid):
            return None
        pts = pose[valid, :2].astype(int)
        return int(pts[:, 0].min()), int(pts[:, 1].min()), \
               int(pts[:, 0].max()), int(pts[:, 1].max())

    @staticmethod
    def get_head_anchor(pose, min_score=0.1):
        """
        Returns (center_x, top_y) for placing a name label above the head.
        Uses nose + eye keypoints (indices 0-4).
        Returns None if no head keypoints are confident.
        """
        head = pose[:5, :]
        valid = head[:, 2] > min_score
        if np.any(valid):
            cx = int(head[valid, 0].mean())
            top_y = int(head[valid, 1].min())
            return cx, top_y
        # Fallback: use any valid keypoint
        valid_all = pose[:, 2] > min_score
        if not np.any(valid_all):
            return None
        cx = int(pose[valid_all, 0].mean())
        top_y = int(pose[valid_all, 1].min())
        return cx, top_y

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, frame):
        h, w = frame.shape[:2]
        scale = self._input_h / h
        img = cv2.resize(frame, None, fx=scale, fy=scale)
        new_h, new_w = img.shape[:2]
        if new_w < self._input_w:
            img = np.pad(img, ((0, 0), (0, self._input_w - new_w), (0, 0)),
                         mode='constant', constant_values=0)
        else:
            img = img[:, :self._input_w]
        scale_xy = np.array([w / new_w, h / new_h], dtype=np.float32)
        img = img.transpose((2, 0, 1))[None].astype(np.float32)
        return img, scale_xy

    def _apply_nms(self, heatmaps):
        """Max-pool NMS equivalent to the opset8 approach but using cv2.dilate."""
        nms = np.zeros_like(heatmaps)
        k = self._nms_kernel
        kernel = np.ones((k, k), np.float32)
        for c in range(heatmaps.shape[1]):
            hm = heatmaps[0, c]
            dilated = cv2.dilate(hm, kernel)
            nms[0, c] = hm * (hm == dilated)
        return nms
