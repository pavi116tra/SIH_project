"""
ByteTrack Multi-Object Tracker implementation with Kalman Filter state estimation
and Hungarian algorithm association for real-time human tracking.
"""

import numpy as np
from enum import Enum
from scipy.optimize import linear_sum_assignment


class TrackState(Enum):
    NEW = 1
    CONFIRMED = 2
    ACTIVE = 3
    TEMPORARILY_LOST = 4
    REMOVED = 5


class KalmanFilterBoundingBox:
    """
    A 2D Kalman Filter for tracking bounding boxes in image space.
    State: [x_center, y_center, area, aspect_ratio, vx, vy, va, vr]
    """

    def __init__(self):
        # State dimension 8 (4 position + 4 velocity), Measurement dimension 4 (position)
        self._motion_mat = np.eye(8, 8)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

        self._update_mat = np.eye(4, 8)

        # Variance weights
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement):
        """Create track from unassociated measurement [x1, y1, x2, y2]."""
        mean_pos = self._bbox_to_z(measurement)
        mean_vel = np.zeros(4)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Predict state mean and covariance forward one time step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot([self._motion_mat, covariance, self._motion_mat.T])
            + motion_cov
        )
        return mean, covariance

    def update(self, mean, covariance, measurement):
        """Update state using observed measurement."""
        z = self._bbox_to_z(measurement)

        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = (
            np.linalg.multi_dot(
                [self._update_mat, covariance, self._update_mat.T]
            )
            + np.diag(np.square(std))
        )

        kalman_gain = np.linalg.multi_dot(
            [
                covariance,
                self._update_mat.T,
                np.linalg.inv(innovation_cov),
            ]
        )
        innovation = z - np.dot(self._update_mat, mean)

        new_mean = mean + np.dot(kalman_gain, innovation)
        new_covariance = covariance - np.linalg.multi_dot(
            [kalman_gain, self._update_mat, covariance]
        )
        return new_mean, new_covariance

    @staticmethod
    def _bbox_to_z(bbox):
        """Convert [x1, y1, x2, y2] to [x_center, y_center, area, aspect_ratio]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h
        r = w / float(h) if h > 0 else 1.0
        return np.array([x, y, s, r])

    @staticmethod
    def z_to_bbox(z):
        """Convert [x_center, y_center, area, aspect_ratio] to [x1, y1, x2, y2]."""
        w = np.sqrt(z[2] * z[3]) if z[2] * z[3] > 0 else 0
        h = z[2] / float(w) if w > 0 else 0
        return np.array(
            [z[0] - w / 2.0, z[1] - h / 2.0, z[0] + w / 2.0, z[1] + h / 2.0]
        )


class Track:
    """Represents a single tracked human object over time."""

    _count = 0

    def __init__(self, bbox, score, class_id=0, class_name="Person"):
        Track._count += 1
        self.track_id = Track._count
        self.display_id = self.track_id
        self.bbox = np.array(bbox, dtype=np.float32)
        self.score = float(score)
        self.class_id = int(class_id)
        self.class_name = str(class_name)

        self.kf = KalmanFilterBoundingBox()
        self.mean, self.covariance = self.kf.initiate(self.bbox)

        self.state = TrackState.NEW
        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.history = []

        # Store center coordinate for trajectory visualization
        center_x = float((bbox[0] + bbox[2]) / 2.0)
        center_y = float((bbox[1] + bbox[3]) / 2.0)
        self.history.append((center_x, center_y))

    @property
    def tlbr(self):
        return self.bbox

    def predict(self):
        """Predict next position using Kalman Filter."""
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self.bbox = self.kf.z_to_bbox(self.mean[:4])

    def update(self, bbox, score):
        """Update track with new matched detection."""
        self.bbox = np.array(bbox, dtype=np.float32)
        self.score = float(score)
        self.mean, self.covariance = self.kf.update(
            self.mean, self.covariance, self.bbox
        )
        self.hits += 1
        self.time_since_update = 0

        if self.state == TrackState.NEW and self.hits >= 2:
            self.state = TrackState.CONFIRMED
        elif self.state in [TrackState.CONFIRMED, TrackState.TEMPORARILY_LOST]:
            self.state = TrackState.ACTIVE

        center_x = float((bbox[0] + bbox[2]) / 2.0)
        center_y = float((bbox[1] + bbox[3]) / 2.0)
        self.history.append((center_x, center_y))
        if len(self.history) > 50:
            self.history.pop(0)

    def mark_lost(self):
        """Mark track as temporarily lost when unassociated."""
        self.state = TrackState.TEMPORARILY_LOST

    def mark_removed(self):
        """Mark track as permanently removed."""
        self.state = TrackState.REMOVED


def calculate_iou(box1, box2):
    """Compute Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / float(union_area) if union_area > 0 else 0.0


def compute_iou_matrix(tracks, detections):
    """Compute IoU cost matrix between tracks and detections."""
    matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(detections):
            matrix[i, j] = calculate_iou(t.bbox, d['bbox'])
    return matrix


class ByteTracker:
    """
    ByteTrack association algorithm for multi-object tracking.
    Matches high-confidence detections first, then low-confidence detections
    to recover partially occluded or blurry targets.
    """

    def __init__(
        self,
        track_high_thresh=0.50,
        track_low_thresh=0.15,
        new_track_thresh=0.60,
        match_thresh=0.80,
        track_buffer=30,
        confirm_frames=3,
    ):
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.confirm_frames = confirm_frames

        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []
        self.frame_id = 0

    @property
    def tracked_stracks(self):
        return self.tracked_tracks

    @property
    def lost_stracks(self):
        return self.lost_tracks

    @property
    def removed_stracks(self):
        return self.removed_tracks

    def update(self, detections):
        """
        Update tracker state with new frame detections.
        detections: list of dicts {'bbox': [x1,y1,x2,y2], 'score': float, 'class_id': int, 'class_name': str}
        """
        self.frame_id += 1

        # 1. Separate detections into high-confidence and low-confidence
        high_dets = [
            d for d in detections if d['score'] >= self.track_high_thresh
        ]
        low_dets = [
            d
            for d in detections
            if self.track_low_thresh <= d['score'] < self.track_high_thresh
        ]

        # 2. Predict positions for all active tracks
        unconfirmed = []
        tracked = []
        for t in self.tracked_tracks:
            t.predict()
            if t.state == TrackState.NEW:
                unconfirmed.append(t)
            else:
                tracked.append(t)

        for t in self.lost_tracks:
            t.predict()

        # Combine active and temporarily lost tracks for matching
        track_pool = tracked + self.lost_tracks

        # 3. First Stage Matching: High-confidence detections with active tracks
        iou_mat = compute_iou_matrix(track_pool, high_dets)
        cost_mat = 1.0 - iou_mat
        matched_tracks_1, matched_dets_1 = [], []

        if cost_mat.size > 0:
            row_ind, col_ind = linear_sum_assignment(cost_mat)
            for r, c in zip(row_ind, col_ind):
                if cost_mat[r, c] < self.match_thresh:
                    matched_tracks_1.append(r)
                    matched_dets_1.append(c)
                    track_pool[r].update(
                        high_dets[c]['bbox'], high_dets[c]['score']
                    )

        # Unmatched tracks and detections from Stage 1
        unmatched_tracks_1 = [
            track_pool[i]
            for i in range(len(track_pool))
            if i not in matched_tracks_1
        ]
        unmatched_high_dets = [
            high_dets[i]
            for i in range(len(high_dets))
            if i not in matched_dets_1
        ]

        # 4. Second Stage Matching: Low-confidence detections with remaining tracks
        iou_mat_2 = compute_iou_matrix(unmatched_tracks_1, low_dets)
        cost_mat_2 = 1.0 - iou_mat_2
        matched_tracks_2, matched_dets_2 = [], []

        if cost_mat_2.size > 0:
            row_ind_2, col_ind_2 = linear_sum_assignment(cost_mat_2)
            for r, c in zip(row_ind_2, col_ind_2):
                if cost_mat_2[r, c] < 0.5:  # Strict threshold for low-conf
                    matched_tracks_2.append(r)
                    matched_dets_2.append(c)
                    unmatched_tracks_1[r].update(
                        low_dets[c]['bbox'], low_dets[c]['score']
                    )

        unmatched_tracks_2 = [
            unmatched_tracks_1[i]
            for i in range(len(unmatched_tracks_1))
            if i not in matched_tracks_2
        ]

        # 5. Third Stage Matching: Unconfirmed new tracks with unmatched high-confidence detections
        iou_mat_3 = compute_iou_matrix(unconfirmed, unmatched_high_dets)
        cost_mat_3 = 1.0 - iou_mat_3
        matched_tracks_3, matched_dets_3 = [], []

        if cost_mat_3.size > 0:
            row_ind_3, col_ind_3 = linear_sum_assignment(cost_mat_3)
            for r, c in zip(row_ind_3, col_ind_3):
                if cost_mat_3[r, c] < self.match_thresh:
                    matched_tracks_3.append(r)
                    matched_dets_3.append(c)
                    unconfirmed[r].update(
                        unmatched_high_dets[c]['bbox'],
                        unmatched_high_dets[c]['score'],
                    )

        unmatched_high_dets_final = [
            unmatched_high_dets[i]
            for i in range(len(unmatched_high_dets))
            if i not in matched_dets_3
        ]
        unmatched_unconfirmed = [
            unconfirmed[i]
            for i in range(len(unconfirmed))
            if i not in matched_tracks_3
        ]

        # 6. Initialize New Tracks for unmatched high-confidence detections
        new_tracks = []
        for d in unmatched_high_dets_final:
            if d['score'] >= self.new_track_thresh:
                track = Track(
                    d['bbox'],
                    d['score'],
                    d.get('class_id', 0),
                    d.get('class_name', 'Person'),
                )
                new_tracks.append(track)

        # 7. Update Track Lists and Life Cycles
        self.tracked_tracks = []
        self.lost_tracks = []

        for t in track_pool + unconfirmed + new_tracks:
            if t in unmatched_unconfirmed:
                t.mark_removed()
                self.removed_tracks.append(t)
                continue

            if t.time_since_update == 0:
                if t.hits >= self.confirm_frames:
                    t.state = TrackState.ACTIVE
                else:
                    t.state = TrackState.CONFIRMED
                self.tracked_tracks.append(t)
            else:
                if t.time_since_update > self.track_buffer:
                    t.mark_removed()
                    self.removed_tracks.append(t)
                else:
                    t.mark_lost()
                    self.lost_tracks.append(t)

        # Filter out removed tracks
        output_tracks = [
            t
            for t in self.tracked_tracks
            if t.state in [TrackState.ACTIVE, TrackState.CONFIRMED]
        ]
        return output_tracks
