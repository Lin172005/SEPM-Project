import cv2
import numpy as np
import time
from collections import deque
from typing import List, Tuple, Dict

class AdvancedAnomalyDetector:
    """
    Advanced anomaly detection system for CCTV surveillance.
    Responsible for detecting various types of anomalies in video frames.
    
    Member 1's Responsibilities:
    1. Implement sophisticated detection algorithms
    2. Optimize detection performance
    3. Add new detection features
    4. Tune detection parameters
    """
    
    def __init__(self):
        # Initialize buffers with deque for better performance
        self.prev_boxes = deque(maxlen=3)  # Increased buffer for better motion tracking
        self.movement_buffer = deque(maxlen=5)  # Increased buffer for movement patterns
        self.motion_history = deque(maxlen=10)  # New buffer for motion history
        
        # Detection parameters - Adjusted thresholds for less sensitivity
        self.violence_threshold = 1.2  # Increased from 0.75 (higher value = less sensitive)
        self.min_violence_frames = 3   # Increased from 2 (requires more consecutive frames)
        self.proximity_threshold = 100  # Increased from 0.8 (requires closer proximity)
        self.movement_threshold = 0.3  # Increased from 0.15 (requires more significant movement)
        self.min_people_for_violence = 2
        self.max_people_for_violence = 4
        
        # Perspective and scale parameters
        self.perspective_zones = [
            {'y_threshold': 0.3, 'scale_factor': 1.5},  # Top zone (far)
            {'y_threshold': 0.6, 'scale_factor': 1.2},  # Middle zone
            {'y_threshold': 1.0, 'scale_factor': 1.0}   # Bottom zone (near)
        ]
        self.min_person_height = 50  # Minimum height in pixels
        self.max_person_height = 400  # Maximum height in pixels
        self.height_ratio_threshold = 3.0  # Max ratio between tallest and shortest person
        
        # Corner view compensation
        self.corner_compensation = True
        self.corner_angle_threshold = 30  # degrees
        self.corner_scale_factor = 1.3
        
        # Motion detection parameters - Adjusted for less sensitivity
        self.fast_motion_threshold = 40  # Increased from 30 (requires faster movement)
        self.motion_persistence = 4  # Increased from 3 (requires longer duration)
        self.motion_history_weight = 0.8  # Increased from 0.7 (more weight on historical motion)
        self.sudden_motion_threshold = 2.5  # Increased from 2.0 (requires more sudden movement)
        
        # Previous frame for motion detection
        self.prev_frame = None
        self.motion_mask = None
        
        # Crowd detection parameters - Optimized
        self.crowd_threshold = 8
        self.density_threshold = 0.35
        self.min_distance = 60
        self.frame_width = 640
        self.frame_height = 480
        self.crowd_zone = None
        
        # Night mode parameters
        self.is_night_mode = False
        self.brightness_threshold = 50  # Average brightness below this value triggers night mode
        self.night_contrast_alpha = 1.5  # Contrast enhancement factor for night mode
        self.night_brightness_beta = 30  # Brightness enhancement factor for night mode
        
        # Performance optimization
        self.skip_frames = 0
        self.process_every_n_frames = 1  # Changed to 1 to catch fast movements
        self.last_crowd_check = 0
        self.crowd_check_interval = 1.0
        self.last_detections = None
        
        # Night detection history
        self.brightness_history = deque(maxlen=10)
        self.last_mode_switch = time.time()
        self.mode_switch_cooldown = 5.0  # Minimum time between mode switches
        
        # Pose detection parameters
        self.pose_threshold = 0.3
        self.min_pose_confidence = 0.5
        self.pose_buffer = deque(maxlen=5)  # Store recent poses
        self.pose_change_threshold = 0.3  # Threshold for significant pose changes
        
        # Bending detection parameters
        self.bending_angle_threshold = 45  # degrees
        self.bending_confidence_threshold = 0.6
        self.bending_buffer = deque(maxlen=3)  # Store recent bending states
        
        # Size validation parameters (adjusted for bending)
        self.min_person_height = 30  # Reduced for bent positions
        self.max_person_height = 400
        self.min_person_width = 20  # Added minimum width
        self.max_person_width = 200
        self.aspect_ratio_range = (0.5, 4.0)  # Wider range for different poses
        
        # Upper-angle view parameters
        self.upper_angle_threshold = 30  # degrees from vertical
        self.upper_angle_compensation = True
        self.upper_angle_scale_factors = {
            'top': 1.8,    # Far objects appear smaller
            'middle': 1.4, # Middle distance
            'bottom': 1.0  # Near objects
        }
        self.upper_angle_height_ratios = {
            'top': 0.6,    # Reduced height ratio for far objects
            'middle': 0.8, # Middle distance
            'bottom': 1.0  # Near objects
        }
        
        # Adjusted size parameters for upper-angle view
        self.min_person_height = 25  # Reduced for upper-angle view
        self.max_person_height = 350
        self.min_person_width = 15
        self.max_person_width = 180
        self.aspect_ratio_range = (0.4, 3.5)  # Adjusted for upper-angle view
        
        self.history = []
    
    def enhance_night_vision(self, frame):
        """Enhance frame for better night vision"""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced_l = clahe.apply(l)
            
            # Merge channels
            enhanced_lab = cv2.merge([enhanced_l, a, b])
            
            # Convert back to BGR
            enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            # Apply additional brightness and contrast adjustment
            enhanced_frame = cv2.convertScaleAbs(enhanced_frame, 
                                               alpha=self.night_contrast_alpha, 
                                               beta=self.night_brightness_beta)
            
            return enhanced_frame
        except Exception as e:
            print(f"Error in night vision enhancement: {e}")
            return frame
    
    def check_lighting_conditions(self, frame):
        """Check if night mode should be activated"""
        try:
            # Calculate average brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            
            # Add to history
            self.brightness_history.append(avg_brightness)
            
            # Check if enough time has passed since last mode switch
            current_time = time.time()
            if current_time - self.last_mode_switch < self.mode_switch_cooldown:
                return self.is_night_mode
            
            # Calculate average brightness from history
            avg_brightness = np.mean(self.brightness_history)
            
            # Determine if mode should change
            should_be_night_mode = avg_brightness < self.brightness_threshold
            
            # Switch mode if necessary
            if should_be_night_mode != self.is_night_mode:
                self.is_night_mode = should_be_night_mode
                self.last_mode_switch = current_time
                
                # Adjust detection parameters for night mode
                if self.is_night_mode:
                    self.violence_threshold *= 0.8  # More sensitive in night mode
                    self.proximity_threshold *= 1.2  # Allow greater distances
                    self.movement_threshold *= 0.8  # More sensitive to movement
                else:
                    # Reset to default values
                    self.violence_threshold = 1.2
                    self.proximity_threshold = 100
                    self.movement_threshold = 0.3
            
            return self.is_night_mode
        except Exception as e:
            print(f"Error checking lighting conditions: {e}")
            return False
    
    def preprocess_frame(self, frame):
        """Preprocess frame based on lighting conditions"""
        try:
            # Check lighting conditions
            is_night = self.check_lighting_conditions(frame)
            
            if is_night:
                # Apply night vision enhancement
                frame = self.enhance_night_vision(frame)
                
                # Add night mode indicator
                cv2.putText(frame, "NIGHT MODE", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return frame
        except Exception as e:
            print(f"Error in frame preprocessing: {e}")
            return frame
    
    def calculate_box_center(self, box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)  # Using multiplication instead of division
    
    def calculate_box_size(self, box):
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)
    
    def calculate_normalized_distance(self, box1, box2):
        # Optimized distance calculation
        c1x, c1y = self.calculate_box_center(box1)
        c2x, c2y = self.calculate_box_center(box2)
        
        # Using squared distance to avoid sqrt when possible
        dist_sq = (c2x - c1x)**2 + (c2y - c1y)**2
        
        # Only calculate sqrt if needed for final normalization
        size1 = self.calculate_box_size(box1)
        size2 = self.calculate_box_size(box2)
        avg_size = (size1 + size2) * 0.5
        
        return np.sqrt(dist_sq / avg_size)
    
    def detect_motion(self, frame):
        """Detect motion between frames using optical flow"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.motion_mask = np.zeros_like(self.prev_frame)
            return None
        
        # Convert current frame to grayscale
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, current_frame, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Calculate motion magnitude
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_score = np.mean(magnitude)
        
        # Update motion history
        self.motion_history.append(motion_score)
        
        # Update previous frame
        self.prev_frame = current_frame
        
        return motion_score
    
    def analyze_movement_patterns(self, current_boxes):
        """Analyze movement patterns between frames"""
        if not self.prev_boxes or not current_boxes:
            return 0.0, []
        
        movement_scores = []
        movement_indicators = []
        
        # Compare with previous frames
        for prev_boxes in self.prev_boxes:
            if len(prev_boxes) != len(current_boxes):
                continue
                
            for curr_box, prev_box in zip(current_boxes, prev_boxes):
                # Calculate center points
                curr_center = self.calculate_box_center(curr_box)
                prev_center = self.calculate_box_center(prev_box)
                
                # Calculate movement distance and speed
                distance = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                 (curr_center[1] - prev_center[1])**2)
                
                # Normalize by box size
                box_size = self.calculate_box_size(curr_box)
                normalized_movement = distance / np.sqrt(box_size)
                
                # Check for sudden movements
                if normalized_movement > self.movement_threshold * self.sudden_motion_threshold:
                    movement_indicators.append("Sudden rapid movement detected")
                
                movement_scores.append(normalized_movement)
        
        # Calculate overall movement score
        if movement_scores:
            # Weight recent movements more heavily
            weighted_scores = [score * (i + 1) / len(movement_scores) 
                            for i, score in enumerate(movement_scores)]
            movement_score = np.mean(weighted_scores)
        else:
            movement_score = 0.0
        
        return movement_score, movement_indicators
    
    def detect_violence(self, frame, detections):
        violence_score = 0.0
        indicators = []
        
        # Skip if not enough or too many people
        if len(detections) < self.min_people_for_violence or len(detections) > self.max_people_for_violence:
            return False, 0.0, []
        
        # Detect overall motion in frame
        motion_score = self.detect_motion(frame)
        if motion_score is not None:
            # Check for high motion
            if motion_score > self.fast_motion_threshold:
                violence_score += 0.3  # Add base score for high motion
                indicators.append(f"High motion detected ({motion_score:.2f})")
        
        # Analyze movement patterns
        movement_score, movement_indicators = self.analyze_movement_patterns(detections)
        if movement_score > 0:
            violence_score += movement_score
            indicators.extend(movement_indicators)
        
        # Check proximity between people
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                box1 = detections[i]
                box2 = detections[j]
                
                # Calculate normalized distance
                distance = self.calculate_normalized_distance(box1, box2)
                
                # If people are very close
                if distance < self.proximity_threshold:
                    proximity_score = 1.0 - (distance / self.proximity_threshold)
                    violence_score += proximity_score
                    
                    # Check for combined proximity and movement
                    if movement_score > self.movement_threshold:
                        violence_score *= 1.5  # Increase score for combined factors
                        indicators.append("Close proximity with rapid movement")
        
        # Update buffers
        self.prev_boxes.append(detections)
        self.movement_buffer.append(violence_score)
        
        # Calculate final violence score with historical context
        if len(self.movement_buffer) >= self.min_violence_frames:
            recent_violence = np.mean(list(self.movement_buffer)[-self.min_violence_frames:])
            if recent_violence > self.violence_threshold:
                return True, recent_violence, indicators
        
        return False, violence_score, indicators
    
    def detect_overcrowding(self, frame, detections):
        current_time = time.time()
        
        # Use cached result if within interval
        if (current_time - self.last_crowd_check) < self.crowd_check_interval:
            return False, None
            
        if len(detections) < self.crowd_threshold:
            return False, None
        
        # Update dimensions and calculate area
        self.frame_width = frame.shape[1]
        self.frame_height = frame.shape[0]
        frame_area = self.frame_width * self.frame_height
        
        # Quick area calculation
        total_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in detections)
        area_density = total_area / frame_area
        
        # Only check proximity if area density is significant
        if area_density > self.density_threshold * 0.5:
            # Calculate centers once
            centers = [(int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)) 
                      for x1, y1, x2, y2 in detections]
            
            # Efficient proximity counting
            proximity_count = sum(1 for i in range(len(centers))
                                for j in range(i + 1, len(centers))
                                if ((centers[i][0] - centers[j][0])**2 + 
                                    (centers[i][1] - centers[j][1])**2) < self.min_distance**2)
            
            max_connections = (len(detections) * (len(detections) - 1)) * 0.5
            proximity_density = proximity_count / max_connections if max_connections > 0 else 0
            
            # Combined score
            density_score = (area_density + proximity_density) * 0.5
            
            if density_score > self.density_threshold:
                # Calculate crowd zone
                min_x = min(x1 for x1, _, _, _ in detections)
                min_y = min(y1 for _, y1, _, _ in detections)
                max_x = max(x2 for _, _, x2, _ in detections)
                max_y = max(y2 for _, _, _, y2 in detections)
                
                padding = 20
                self.crowd_zone = (
                    max(0, min_x - padding),
                    max(0, min_y - padding),
                    min(self.frame_width, max_x + padding),
                    min(self.frame_height, max_y + padding)
                )
                
                self.last_crowd_check = current_time
                return True, f"Overcrowding: {len(detections)} people (Density: {density_score:.2f})"
        
        self.last_crowd_check = current_time
        return False, None
    
    def adjust_for_perspective(self, box, frame_height):
        """Adjust detection based on perspective zone"""
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        box_center_y = (y1 + y2) / 2
        relative_y = box_center_y / frame_height
        
        # Find appropriate scale factor based on zone
        scale_factor = 1.0
        for zone in self.perspective_zones:
            if relative_y <= zone['y_threshold']:
                scale_factor = zone['scale_factor']
                break
        
        # Apply scale factor to box dimensions
        box_width = x2 - x1
        new_width = box_width * scale_factor
        new_height = box_height * scale_factor
        
        # Adjust box while maintaining center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        new_x1 = int(center_x - new_width / 2)
        new_x2 = int(center_x + new_width / 2)
        new_y1 = int(center_y - new_height / 2)
        new_y2 = int(center_y + new_height / 2)
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def detect_pose(self, box, frame):
        """Detect pose and determine if person is bending"""
        x1, y1, x2, y2 = box
        person_roi = frame[y1:y2, x1:x2]
        
        if person_roi.size == 0:
            return False, 0.0
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, 0.0
        
        # Get the largest contour (person's silhouette)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a rectangle to the contour
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Calculate bending confidence
        height = y2 - y1
        width = x2 - x1
        aspect_ratio = height / width if width > 0 else 0
        
        # Determine if person is bending based on angle and aspect ratio
        is_bending = abs(angle) > self.bending_angle_threshold or aspect_ratio < 1.0
        bending_confidence = min(1.0, abs(angle) / 90.0)  # Normalize confidence
        
        return is_bending, bending_confidence
    
    def adjust_for_upper_angle(self, box, frame_height):
        """Adjust detection based on upper-angle view"""
        x1, y1, x2, y2 = box
        box_height = y2 - y1
        box_center_y = (y1 + y2) / 2
        relative_y = box_center_y / frame_height
        
        # Determine zone based on vertical position
        if relative_y < 0.33:
            zone = 'top'
        elif relative_y < 0.66:
            zone = 'middle'
        else:
            zone = 'bottom'
        
        # Get scale factors for this zone
        scale_factor = self.upper_angle_scale_factors[zone]
        height_ratio = self.upper_angle_height_ratios[zone]
        
        # Apply scale factor to box dimensions
        box_width = x2 - x1
        new_width = box_width * scale_factor
        new_height = box_height * scale_factor * height_ratio
        
        # Adjust box while maintaining center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        new_x1 = int(center_x - new_width / 2)
        new_x2 = int(center_x + new_width / 2)
        new_y1 = int(center_y - new_height / 2)
        new_y2 = int(center_y + new_height / 2)
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def validate_detection(self, box, frame_height, all_boxes, frame):
        """Enhanced validation for upper-angle view"""
        x1, y1, x2, y2 = box
        height = y2 - y1
        width = x2 - x1
        aspect_ratio = height / width if width > 0 else 0
        
        # Check if person is bending
        is_bending, bending_confidence = self.detect_pose(box, frame)
        
        # Get relative position in frame
        relative_y = (y1 + y2) / 2 / frame_height
        
        # Adjust thresholds based on position in frame
        if relative_y < 0.33:  # Top of frame (farther)
            min_height = self.min_person_height * 0.7
            max_height = self.max_person_height * 0.9
            min_width = self.min_person_width * 0.8
            max_width = self.max_person_width * 0.9
        elif relative_y < 0.66:  # Middle of frame
            min_height = self.min_person_height * 0.85
            max_height = self.max_person_height * 0.95
            min_width = self.min_person_width * 0.9
            max_width = self.max_person_width * 0.95
        else:  # Bottom of frame (closer)
            min_height = self.min_person_height
            max_height = self.max_person_height
            min_width = self.min_person_width
            max_width = self.max_person_width
        
        # Further adjust for bending
        if is_bending and bending_confidence > self.bending_confidence_threshold:
            min_height *= 0.7
            max_height *= 1.2
        
        # Size validation with adjusted thresholds
        if height < min_height or height > max_height:
            return False
        
        # Width validation
        if width < min_width or width > max_width:
            return False
        
        # Aspect ratio check with position-based adjustment
        min_ratio, max_ratio = self.aspect_ratio_range
        if relative_y < 0.33:  # Top of frame
            min_ratio *= 0.7
            max_ratio *= 0.9
        elif relative_y < 0.66:  # Middle of frame
            min_ratio *= 0.8
            max_ratio *= 0.95
        
        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
            return False
        
        # Compare with other detections
        if all_boxes:
            heights = [b[3] - b[1] for b in all_boxes]
            max_height = max(heights)
            min_height = min(heights)
            
            # Adjust height ratio threshold based on position
            if relative_y < 0.33:
                height_ratio_threshold = self.height_ratio_threshold * 1.8
            elif relative_y < 0.66:
                height_ratio_threshold = self.height_ratio_threshold * 1.4
            else:
                height_ratio_threshold = self.height_ratio_threshold
            
            # Further adjust for bending
            if is_bending:
                height_ratio_threshold *= 1.5
            
            if max_height / min_height > height_ratio_threshold:
                return False
        
        return True
    
    def process_detections(self, detections, frame):
        """Process and filter detections considering upper-angle view"""
        if not detections:
            return []
        
        frame_height = frame.shape[0]
        valid_detections = []
        
        # First pass: collect all valid detections
        for box in detections:
            if self.validate_detection(box, frame_height, detections, frame):
                # Apply both perspective and upper-angle adjustments
                perspective_adjusted = self.adjust_for_perspective(box, frame_height)
                final_adjusted = self.adjust_for_upper_angle(perspective_adjusted, frame_height)
                valid_detections.append(final_adjusted)
        
        return valid_detections
    
    def detect_anomalies(self, frame, detections):
        # Skip frames for performance
        self.skip_frames = (self.skip_frames + 1) % self.process_every_n_frames
        if self.skip_frames != 0:
            return []
            
        anomalies = []
        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)
            
            # Process detections with upper-angle consideration
            processed_detections = self.process_detections(detections, frame)
            
            # Violence detection with processed detections
            is_violence, score, indicators = self.detect_violence(processed_frame, processed_detections)
            if is_violence:
                anomalies.append(("Violence", f"Violence detected ({score:.2f}) - {', '.join(indicators)}"))
            
            # Crowd detection with processed detections
            is_crowd, crowd_msg = self.detect_overcrowding(processed_frame, processed_detections)
            if is_crowd:
                anomalies.append(("Overcrowding", crowd_msg))
                if self.crowd_zone:
                    x1, y1, x2, y2 = self.crowd_zone
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, "Crowded Area", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Draw perspective zones for debugging (optional)
            self.draw_perspective_zones(frame)
            
            # Draw pose information (optional)
            for box in processed_detections:
                is_bending, confidence = self.detect_pose(box, frame)
                if is_bending:
                    x1, y1, x2, y2 = box
                    cv2.putText(frame, f"Bending ({confidence:.2f})", 
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 255), 1)
            
            # Draw upper-angle zones (optional)
            self.draw_upper_angle_zones(frame)
            
            # Update frame with enhanced version if in night mode
            if self.is_night_mode:
                frame[:] = processed_frame[:]
        
        except Exception as e:
            print(f"Detection error: {e}")
        
        return anomalies
    
    def draw_perspective_zones(self, frame):
        """Draw perspective zones for visualization"""
        height, width = frame.shape[:2]
        colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]  # Green, Yellow, Red
        
        for i, zone in enumerate(self.perspective_zones):
            y = int(height * zone['y_threshold'])
            cv2.line(frame, (0, y), (width, y), colors[i], 1)
            cv2.putText(frame, f"Zone {i+1} (x{zone['scale_factor']})", 
                       (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
    
    def draw_upper_angle_zones(self, frame):
        """Draw upper-angle zones for visualization"""
        height, width = frame.shape[:2]
        colors = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]  # Green, Yellow, Red
        
        # Draw zone boundaries
        for i, (zone, y_threshold) in enumerate([('top', 0.33), ('middle', 0.66), ('bottom', 1.0)]):
            y = int(height * y_threshold)
            cv2.line(frame, (0, y), (width, y), colors[i], 1)
            scale_factor = self.upper_angle_scale_factors[zone]
            height_ratio = self.upper_angle_height_ratios[zone]
            cv2.putText(frame, f"{zone.title()} (x{scale_factor:.1f}, h{height_ratio:.1f})", 
                       (10, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)
    
    def update_history(self, frame: np.ndarray, detections: List[List[int]]):
        """Update detection history for temporal analysis"""
        # TODO: Member 1 - Implement history tracking
        pass 