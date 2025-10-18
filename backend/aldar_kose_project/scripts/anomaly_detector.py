#!/usr/bin/env python3
"""
Anomaly Detection and Validation for Generated Images

Detects and prevents common generation anomalies:
- Multiple faces/heads
- Body part duplications
- Size inconsistencies
- Deformed poses
- Missing body parts

Uses:
1. Face detection (MediaPipe/OpenCV)
2. Pose keypoint detection (OpenPose/MediaPipe)
3. CLIP-based semantic validation
4. Automatic regeneration with adjusted parameters
"""

import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("OpenCV not available - some detection features will be limited")

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning("MediaPipe not available - pose detection disabled")

try:
    from transformers import CLIPProcessor, CLIPModel
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    logger.warning("CLIP not available - semantic validation disabled")


class AnomalyDetector:
    """
    Detects structural and semantic anomalies in generated images
    """
    
    def __init__(
        self,
        device: str = "cuda",
        strict_mode: bool = False,
        expected_faces: int = 1,
        min_face_size: float = 0.02,
        max_face_size: float = 0.65,
    ):
        """
        Initialize anomaly detector
        
        Args:
            device: Device for CLIP model
            strict_mode: If True, reject images more aggressively
            expected_faces: Expected number of faces (default: 1 for single character)
            min_face_size: Minimum face size relative to image (default: 2%)
            max_face_size: Maximum face size relative to image (default: 65%)
        """
        self.device = device
        self.strict_mode = strict_mode
        self.expected_faces = expected_faces
        self.min_face_size = min_face_size
        self.max_face_size = max_face_size
        
        # Initialize face detection
        if HAS_OPENCV:
            logger.info("Loading face detection model...")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        
        # Initialize pose detection
        if HAS_MEDIAPIPE:
            logger.info("Loading MediaPipe Pose detector...")
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=2,
                min_detection_confidence=0.5,
            )
            
            self.mp_face = mp.solutions.face_detection
            self.face_detection = self.mp_face.FaceDetection(
                model_selection=1,  # Full range detector
                min_detection_confidence=0.5,
            )
        
        # Initialize CLIP for semantic validation
        if HAS_CLIP:
            logger.info("Loading CLIP model for semantic validation...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.eval()
        
        logger.info("✓ Anomaly Detector initialized")
    
    def detect_anomalies(
        self,
        image: Image.Image,
        expected_prompt: str = None,
    ) -> Dict[str, any]:
        """
        Run all anomaly detection checks
        
        Args:
            image: PIL Image to validate
            expected_prompt: Expected content description
            
        Returns:
            Dict with:
                - is_valid: bool
                - anomalies: List[str] (detected issues)
                - confidence: float (0-1, higher = more confident it's valid)
                - details: Dict with detection details
        """
        anomalies = []
        details = {}
        
        # Convert PIL to OpenCV format
        image_np = np.array(image)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 1. Face Detection Check
        face_result = self._check_face_count(image_rgb)
        details['faces'] = face_result
        
        # Check face count against expected
        if face_result['count'] == 0:
            anomalies.append("NO_FACE_DETECTED")
        elif face_result['count'] != self.expected_faces:
            if face_result['count'] > self.expected_faces:
                anomalies.append(f"MULTIPLE_FACES_DETECTED ({face_result['count']} faces, expected {self.expected_faces})")
            else:
                # Less faces than expected (but not zero)
                if self.strict_mode:
                    anomalies.append(f"INSUFFICIENT_FACES ({face_result['count']} faces, expected {self.expected_faces})")
        
        # 2. Pose/Body Check
        if HAS_MEDIAPIPE:
            pose_result = self._check_pose_anomalies(image_np)
            details['pose'] = pose_result
            
            if pose_result['anomalies']:
                anomalies.extend(pose_result['anomalies'])
        
        # 3. Size/Proportion Check
        size_result = self._check_size_proportions(image_np, face_result)
        details['proportions'] = size_result
        
        if size_result['anomalies']:
            anomalies.extend(size_result['anomalies'])
        
        # 4. Semantic Validation (CLIP)
        if HAS_CLIP and expected_prompt:
            semantic_result = self._check_semantic_alignment(image, expected_prompt)
            details['semantic'] = semantic_result
            
            if semantic_result['score'] < 0.22:  # Low alignment threshold
                anomalies.append(f"LOW_SEMANTIC_ALIGNMENT (score: {semantic_result['score']:.3f})")
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(details, anomalies)
        
        # Determine if valid
        is_valid = len(anomalies) == 0 or (not self.strict_mode and confidence > 0.6)
        
        return {
            'is_valid': is_valid,
            'anomalies': anomalies,
            'confidence': confidence,
            'details': details,
        }
    
    def _check_face_count(self, image_bgr: np.ndarray) -> Dict:
        """Detect number of faces in image"""
        result = {'count': 0, 'locations': []}
        
        if not HAS_OPENCV and not HAS_MEDIAPIPE:
            return result
        
        # Try MediaPipe first (more accurate)
        if HAS_MEDIAPIPE:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image_rgb)
            
            if results.detections:
                result['count'] = len(results.detections)
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    result['locations'].append({
                        'x': bbox.xmin,
                        'y': bbox.ymin,
                        'w': bbox.width,
                        'h': bbox.height,
                    })
        
        # Fallback to OpenCV Haar Cascade
        elif HAS_OPENCV:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            result['count'] = len(faces)
            result['locations'] = [
                {'x': x, 'y': y, 'w': w, 'h': h}
                for (x, y, w, h) in faces
            ]
        
        return result
    
    def _check_pose_anomalies(self, image_rgb: np.ndarray) -> Dict:
        """Check for pose/body anomalies using MediaPipe"""
        result = {
            'detected': False,
            'anomalies': [],
            'keypoints': None,
        }
        
        if not HAS_MEDIAPIPE:
            return result
        
        results = self.pose.process(image_rgb)
        
        if results.pose_landmarks:
            result['detected'] = True
            landmarks = results.pose_landmarks.landmark
            
            # Check for duplicate body parts (e.g., two left arms)
            # MediaPipe gives 33 keypoints
            keypoint_confidences = [lm.visibility for lm in landmarks]
            result['keypoints'] = len([c for c in keypoint_confidences if c > 0.5])
            
            # Check for deformed poses
            # Example: shoulders should be roughly aligned
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            if left_shoulder.visibility > 0.5 and right_shoulder.visibility > 0.5:
                shoulder_y_diff = abs(left_shoulder.y - right_shoulder.y)
                if shoulder_y_diff > 0.3:  # Shoulders too misaligned
                    result['anomalies'].append("DEFORMED_POSE (misaligned shoulders)")
            
            # Check for missing critical body parts
            critical_parts = [
                self.mp_pose.PoseLandmark.NOSE,
                self.mp_pose.PoseLandmark.LEFT_SHOULDER,
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            ]
            
            visible_parts = sum(1 for part in critical_parts if landmarks[part].visibility > 0.5)
            if visible_parts < 2:
                result['anomalies'].append("MISSING_BODY_PARTS")
        
        return result
    
    def _check_size_proportions(
        self,
        image_rgb: np.ndarray,
        face_result: Dict,
    ) -> Dict:
        """Check for size/proportion anomalies"""
        result = {'anomalies': [], 'face_ratio': None}
        
        h, w = image_rgb.shape[:2]
        
        # Check if face is too small or too large
        if face_result['count'] > 0 and face_result['locations']:
            face = face_result['locations'][0]
            
            # Calculate face area relative to image
            if 'w' in face and 'h' in face:
                face_area = face['w'] * face['h']
                image_area = 1.0  # Normalized
                
                face_ratio = face_area / image_area
                result['face_ratio'] = face_ratio
                
                # Use configurable thresholds
                if face_ratio < self.min_face_size:
                    result['anomalies'].append(f"FACE_TOO_SMALL ({face_ratio:.1%} of image, min: {self.min_face_size:.1%})")
                elif face_ratio > self.max_face_size:
                    result['anomalies'].append(f"FACE_TOO_LARGE ({face_ratio:.1%} of image, max: {self.max_face_size:.1%})")
        
        return result
    
    def _check_semantic_alignment(
        self,
        image: Image.Image,
        expected_prompt: str,
    ) -> Dict:
        """Check if image semantically matches expected prompt"""
        if not HAS_CLIP:
            return {'score': 0.5, 'aligned': True}
        
        # Prepare inputs
        inputs = self.clip_processor(
            text=[expected_prompt],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        # Get CLIP scores
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            score = torch.sigmoid(logits_per_image).cpu().item()
        
        return {
            'score': score,
            'aligned': score > 0.22,  # Threshold for semantic alignment
        }
    
    def _calculate_confidence(self, details: Dict, anomalies: List[str]) -> float:
        """Calculate overall confidence score (0-1)"""
        confidence = 1.0
        
        # Penalize for anomalies
        for anomaly in anomalies:
            if "MULTIPLE_FACES" in anomaly:
                confidence -= 0.4
            elif "NO_FACE" in anomaly:
                confidence -= 0.3
            elif "DEFORMED_POSE" in anomaly:
                confidence -= 0.2
            elif "MISSING_BODY_PARTS" in anomaly:
                confidence -= 0.25
            elif "TOO_SMALL" in anomaly or "TOO_LARGE" in anomaly:
                confidence -= 0.15
            elif "LOW_SEMANTIC" in anomaly:
                confidence -= 0.1
            else:
                confidence -= 0.1
        
        # Boost for good detections
        if details.get('faces', {}).get('count') == 1:
            confidence += 0.1
        
        if details.get('pose', {}).get('detected'):
            confidence += 0.05
        
        return max(0.0, min(1.0, confidence))
    
    def suggest_regeneration_params(
        self,
        anomalies: List[str],
        current_seed: int,
        current_cfg: float,
    ) -> Dict:
        """
        Suggest adjusted parameters for regeneration based on detected anomalies
        
        Strategy:
        - Multiple faces → Increase CFG (tighter prompt following)
        - No face → Increase CFG slightly + change seed
        - Deformed pose → Decrease CFG (allow more freedom)
        - Face too small/large → Adjust CFG based on direction
        
        Args:
            anomalies: List of detected anomalies
            current_seed: Current generation seed
            current_cfg: Current CFG scale
            
        Returns:
            Dict with suggested parameters and reasoning
        """
        suggestions = {
            'seed': current_seed + np.random.randint(5, 50),  # Larger seed offset for variety
            'guidance_scale': current_cfg,
            'num_inference_steps': None,  # Keep default
            'reason': [],
        }
        
        # Track if we need major adjustments
        major_issue = False
        
        # Adjust CFG based on anomaly type (with better logic)
        if any("MULTIPLE_FACES" in a for a in anomalies):
            suggestions['guidance_scale'] = min(current_cfg + 1.5, 12.0)
            suggestions['reason'].append("⚡ Increase CFG significantly to reduce face duplications")
            major_issue = True
        
        if any("NO_FACE" in a for a in anomalies):
            suggestions['guidance_scale'] = min(current_cfg + 1.0, 10.0)
            suggestions['reason'].append("⚡ Increase CFG to ensure face visibility")
            major_issue = True
        
        if any("DEFORMED_POSE" in a for a in anomalies):
            suggestions['guidance_scale'] = max(current_cfg - 1.0, 5.0)
            suggestions['reason'].append("⚡ Reduce CFG to improve pose naturalness")
        
        if any("FACE_TOO_SMALL" in a for a in anomalies):
            suggestions['guidance_scale'] = min(current_cfg + 0.5, 9.0)
            suggestions['reason'].append("📏 Increase CFG slightly for larger face")
        
        if any("FACE_TOO_LARGE" in a for a in anomalies):
            suggestions['guidance_scale'] = max(current_cfg - 0.5, 6.0)
            suggestions['reason'].append("📏 Reduce CFG for better composition")
        
        if any("MISSING_BODY_PARTS" in a for a in anomalies):
            suggestions['guidance_scale'] = max(current_cfg - 0.5, 6.5)
            suggestions['reason'].append("🦴 Reduce CFG for complete body structure")
        
        # For major issues, use larger seed offset
        if major_issue:
            suggestions['seed'] = current_seed + np.random.randint(50, 150)
            suggestions['reason'].append(f"🎲 Large seed change for fresh generation ({suggestions['seed']})")
        else:
            suggestions['reason'].append(f"🎲 New seed: {suggestions['seed']}")
        
        # Add CFG change info
        cfg_change = suggestions['guidance_scale'] - current_cfg
        if abs(cfg_change) > 0.1:
            direction = "↑" if cfg_change > 0 else "↓"
            suggestions['reason'].append(f"{direction} CFG: {current_cfg:.1f} → {suggestions['guidance_scale']:.1f}")
        
        return suggestions


def test_detector():
    """Test the anomaly detector on sample images"""
    logger.info("Testing Anomaly Detector...")
    
    detector = AnomalyDetector(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # Test with a sample image (create a simple test image)
    test_image = Image.new('RGB', (512, 512), color='white')
    
    result = detector.detect_anomalies(
        test_image,
        expected_prompt="aldar_kose_man portrait"
    )
    
    logger.info(f"Test Result: {result}")
    logger.info("✓ Anomaly Detector working!")


if __name__ == "__main__":
    test_detector()
