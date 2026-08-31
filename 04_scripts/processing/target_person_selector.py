"""
T5.C.2 — Target Person Selector Module
Robustly selects the seated participant from all candidate detections in multi-person laboratory settings.
"""
import json
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class TargetPersonSelector:
    def __init__(
        self,
        min_area_ratio: float = 0.05,
        target_area_ref: float = 0.22,
        min_torso_joints: int = 2,
        min_composite_score: float = 0.22,
        w_area: float = 0.35,
        w_center: float = 0.30,
        w_torso: float = 0.25,
        w_kpt_conf: float = 0.10
    ):
        self.min_area_ratio = min_area_ratio
        self.target_area_ref = target_area_ref
        self.min_torso_joints = min_torso_joints
        self.min_composite_score = min_composite_score
        
        # Weights
        self.w_area = w_area
        self.w_center = w_center
        self.w_torso = w_torso
        self.w_kpt_conf = w_kpt_conf

    def select_target(
        self,
        boxes_xyxy: np.ndarray,
        boxes_conf: np.ndarray,
        kpts_data: np.ndarray,
        frame_shape: Tuple[int, int],
        view_role: str = "frontal"
    ) -> Dict[str, Any]:
        """
        Selects the best candidate matching the seated participant prior.
        
        Args:
            boxes_xyxy: [N, 4] bounding boxes
            boxes_conf: [N] box detection confidences
            kpts_data: [N, 17, 3] keypoints (x, y, conf)
            frame_shape: (height, width)
            view_role: 'frontal' or 'lateral'
            
        Returns:
            Dict containing selected person keypoints, bbox, scores, and selection metadata.
        """
        h, w = frame_shape[:2]
        frame_area = float(w * h)
        frame_cx, frame_cy = w / 2.0, h / 2.0
        
        num_cands = len(boxes_xyxy)
        if num_cands == 0:
            return {
                "has_target": False,
                "selected_cand_idx": -1,
                "composite_score": 0.0,
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "keypoints": np.zeros((17, 2), dtype=np.float32).tolist(),
                "confidences": np.zeros((17,), dtype=np.float32).tolist(),
                "candidates_count": 0,
                "decision_reason": "No candidates detected in frame"
            }

        candidate_scores = []
        for i in range(num_cands):
            bx = boxes_xyxy[i]
            bc = float(boxes_conf[i])
            bw = max(1.0, bx[2] - bx[0])
            bh = max(1.0, bx[3] - bx[1])
            b_area = bw * bh
            b_area_ratio = b_area / frame_area
            bcx = (bx[0] + bx[2]) / 2.0
            bcy = (bx[1] + bx[3]) / 2.0
            
            # 1. Area score
            area_score = min(1.0, b_area_ratio / self.target_area_ref)
            if b_area_ratio < self.min_area_ratio:
                area_score *= 0.2  # Heavy penalty for tiny background persons
                
            # 2. Center proximity score
            # In seated posture, subject torso & head is centered horizontally
            dist_x = abs(bcx - frame_cx) / (w / 2.0)
            dist_y = abs(bcy - frame_cy) / (h / 2.0)
            dist_center = np.sqrt(dist_x**2 + dist_y**2)
            center_score = max(0.0, 1.0 - dist_center)
            
            # 3. Torso completeness
            kpts_xy = kpts_data[i, :, :2]
            kpts_c = kpts_data[i, :, 2]
            
            # Torso keypoints: 5=L_shoulder, 6=R_shoulder, 11=L_hip, 12=R_hip
            torso_joints_present = sum(1 for j_idx in [5, 6, 11, 12] if kpts_c[j_idx] >= 0.25)
            torso_score = torso_joints_present / 4.0
            
            # 4. Keypoint confidence
            mean_kpt_conf = float(np.mean(kpts_c))
            
            # Composite score
            composite_score = (
                self.w_area * area_score +
                self.w_center * center_score +
                self.w_torso * torso_score +
                self.w_kpt_conf * mean_kpt_conf
            )
            
            candidate_scores.append({
                "cand_idx": i,
                "bbox": bx.tolist(),
                "box_conf": bc,
                "area_ratio": float(b_area_ratio),
                "area_score": float(area_score),
                "center_score": float(center_score),
                "torso_score": float(torso_score),
                "torso_joints_present": int(torso_joints_present),
                "mean_kpt_conf": float(mean_kpt_conf),
                "composite_score": float(composite_score),
                "keypoints": kpts_xy.tolist(),
                "confidences": kpts_c.tolist()
            })

        # Rank candidates by composite score
        candidate_scores.sort(key=lambda x: x["composite_score"], reverse=True)
        best = candidate_scores[0]
        
        if best["composite_score"] < self.min_composite_score:
            return {
                "has_target": False,
                "selected_cand_idx": -1,
                "composite_score": best["composite_score"],
                "bbox": [0.0, 0.0, 0.0, 0.0],
                "keypoints": np.zeros((17, 2), dtype=np.float32).tolist(),
                "confidences": np.zeros((17,), dtype=np.float32).tolist(),
                "candidates_count": num_cands,
                "all_candidates": candidate_scores,
                "decision_reason": f"Top candidate composite score ({best['composite_score']:.3f}) below threshold ({self.min_composite_score})"
            }

        return {
            "has_target": True,
            "selected_cand_idx": best["cand_idx"],
            "composite_score": best["composite_score"],
            "bbox": best["bbox"],
            "keypoints": best["keypoints"],
            "confidences": best["confidences"],
            "area_ratio": best["area_ratio"],
            "torso_joints_present": best["torso_joints_present"],
            "mean_kpt_conf": best["mean_kpt_conf"],
            "candidates_count": num_cands,
            "all_candidates": candidate_scores,
            "decision_reason": f"Target selected with composite score {best['composite_score']:.3f} (Area={best['area_score']:.2f}, Center={best['center_score']:.2f}, Torso={best['torso_score']:.2f})"
        }
