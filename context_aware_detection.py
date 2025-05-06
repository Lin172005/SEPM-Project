import cv2
import numpy as np
from collections import deque
import time
from datetime import datetime

class ContextAwareDetector:
    def __init__(self):
        # Scene context parameters
        self.scene_type = None  # 'indoor', 'outdoor', 'public_space', 'private_space'
        self.time_of_day = None  # 'day', 'night', 'dawn', 'dusk'
        self.expected_activity_level = None  # 'low', 'medium', 'high'
        
        # Historical context
        self.activity_history = deque(maxlen=100)  # Store recent activity levels
        self.normal_behavior_patterns = {}  # Store learned normal patterns
        self.abnormal_events = deque(maxlen=50)  # Store recent abnormal events
        
        # Contextual thresholds
        self.context_thresholds = {
            'indoor': {
                'movement': 0.3,
                'proximity': 0.8,
                'activity': 0.4
            },
            'outdoor': {
                'movement': 0.5,
                'proximity': 0.6,
                'activity': 0.6
            }
        }
        
        # Scene understanding parameters
        self.scene_objects = []  # List of detected objects in scene
        self.scene_layout = None  # Understanding of scene layout
        self.expected_objects = []  # Objects expected in this scene
        
        # Temporal context
        self.time_based_patterns = {
            'morning': {'activity_level': 'medium', 'expected_objects': ['person', 'vehicle']},
            'afternoon': {'activity_level': 'high', 'expected_objects': ['person', 'vehicle']},
            'evening': {'activity_level': 'medium', 'expected_objects': ['person']},
            'night': {'activity_level': 'low', 'expected_objects': ['person']}
        }
        
        # Social context
        self.group_dynamics = {
            'normal_group_size': 2,
            'max_normal_group_size': 5,
            'min_social_distance': 0.5
        }
        
        # Initialize scene analysis
        self.last_scene_analysis = time.time()
        self.scene_analysis_interval = 5.0  # seconds
        
        # Violence detection parameters
        self.movement_history = deque(maxlen=5)  # Store recent movement patterns
        self.violence_thresholds = {
            'sudden_movement': 0.4,  # Threshold for sudden movement detection
            'aggressive_movement': 0.6,  # Threshold for aggressive movement
            'rapid_direction_change': 0.5,  # Threshold for rapid direction changes
            'min_violence_frames': 3  # Minimum consecutive frames for violence detection
        }
        
        # Movement tracking
        self.prev_positions = {}  # Store previous positions of detected people
        self.movement_vectors = {}  # Store movement vectors
        self.violence_scores = {}  # Store violence scores per person
        
    def analyze_scene_context(self, frame):
        """Analyze the overall scene context"""
        current_time = time.time()
        if current_time - self.last_scene_analysis < self.scene_analysis_interval:
            return
            
        # Determine time of day
        hour = datetime.now().hour
        if 5 <= hour < 12:
            self.time_of_day = 'morning'
        elif 12 <= hour < 17:
            self.time_of_day = 'afternoon'
        elif 17 <= hour < 21:
            self.time_of_day = 'evening'
        else:
            self.time_of_day = 'night'
            
        # Analyze scene type based on visual features
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Determine if indoor or outdoor based on brightness and color distribution
        if brightness < 100:
            self.scene_type = 'indoor'
        else:
            self.scene_type = 'outdoor'
            
        # Update expected activity level based on time and scene
        self.expected_activity_level = self.time_based_patterns[self.time_of_day]['activity_level']
        
        self.last_scene_analysis = current_time
        
    def understand_social_context(self, detections):
        """Analyze social interactions and group dynamics"""
        if not detections:
            return None
            
        # Calculate group sizes and distances
        group_sizes = []
        social_distances = []
        
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                # Calculate distance between people
                distance = self._calculate_distance(detections[i], detections[j])
                social_distances.append(distance)
                
                # If people are close, consider them in a group
                if distance < self.group_dynamics['min_social_distance']:
                    group_sizes.append(2)  # At least 2 people in a group
                    
        # Analyze group dynamics
        if group_sizes:
            avg_group_size = sum(group_sizes) / len(group_sizes)
            if avg_group_size > self.group_dynamics['max_normal_group_size']:
                return 'large_group'
            elif avg_group_size > self.group_dynamics['normal_group_size']:
                return 'medium_group'
            else:
                return 'small_group'
                
        return 'individuals'
        
    def analyze_movement(self, detections):
        """Analyze movement patterns for potential violence"""
        current_positions = {}
        current_movement = {}
        violence_indicators = []
        
        # Calculate current positions and movement
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = box
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            current_positions[i] = center
            
            if i in self.prev_positions:
                prev_center = self.prev_positions[i]
                movement = np.sqrt((center[0] - prev_center[0])**2 + (center[1] - prev_center[1])**2)
                current_movement[i] = movement
                
                # Calculate movement vector
                vector = (center[0] - prev_center[0], center[1] - prev_center[1])
                self.movement_vectors[i] = vector
                
                # Update violence score
                if i not in self.violence_scores:
                    self.violence_scores[i] = deque(maxlen=self.violence_thresholds['min_violence_frames'])
                
                # Calculate movement metrics
                speed = movement
                direction_change = 0
                if i in self.movement_vectors and len(self.movement_vectors[i]) > 1:
                    prev_vector = self.movement_vectors[i]
                    direction_change = np.arccos(np.dot(vector, prev_vector) / 
                                              (np.linalg.norm(vector) * np.linalg.norm(prev_vector)))
                
                # Calculate violence score based on multiple factors
                violence_score = (
                    speed * 0.4 +  # Speed component
                    direction_change * 0.3 +  # Direction change component
                    (1 if speed > self.violence_thresholds['sudden_movement'] else 0) * 0.3  # Sudden movement component
                )
                
                self.violence_scores[i].append(violence_score)
                
                # Check for violence indicators
                if len(self.violence_scores[i]) >= self.violence_thresholds['min_violence_frames']:
                    avg_score = sum(self.violence_scores[i]) / len(self.violence_scores[i])
                    if avg_score > self.violence_thresholds['aggressive_movement']:
                        violence_indicators.append(('violent_movement', f'Person {i+1} showing aggressive movement'))
        
        # Update previous positions
        self.prev_positions = current_positions
        
        return violence_indicators
    
    def detect_contextual_anomalies(self, frame, detections):
        """Detect anomalies considering all contextual factors"""
        self.analyze_scene_context(frame)
        
        anomalies = []
        
        # Get current context
        current_context = {
            'scene_type': self.scene_type,
            'time_of_day': self.time_of_day,
            'activity_level': self.expected_activity_level
        }
        
        # Analyze social context
        social_context = self.understand_social_context(detections)
        
        # Analyze movement patterns for violence
        violence_indicators = self.analyze_movement(detections)
        anomalies.extend(violence_indicators)
        
        # Check for context-specific anomalies
        if self.scene_type == 'indoor':
            # Indoor-specific checks
            if social_context == 'large_group':
                anomalies.append(('crowding', 'Unusual large group in indoor space'))
                
        elif self.scene_type == 'outdoor':
            # Outdoor-specific checks
            if social_context == 'large_group' and self.time_of_day == 'night':
                anomalies.append(('suspicious_gathering', 'Large group gathering at night'))
                
        # Time-based anomaly detection
        if self.time_of_day == 'night':
            if len(detections) > 5:  # More people than expected at night
                anomalies.append(('unusual_activity', 'High activity level during night'))
                
        # Activity level anomaly
        current_activity = len(detections)
        self.activity_history.append(current_activity)
        
        if len(self.activity_history) > 10:
            avg_activity = sum(self.activity_history) / len(self.activity_history)
            if current_activity > avg_activity * 2:
                anomalies.append(('unusual_activity', 'Significantly higher than normal activity level'))
                
        # Social distancing anomaly
        if social_context == 'individuals':
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    distance = self._calculate_distance(detections[i], detections[j])
                    if distance < self.group_dynamics['min_social_distance']:
                        anomalies.append(('proximity', 'People too close together'))
                        
        return anomalies
        
    def _calculate_distance(self, box1, box2):
        """Calculate normalized distance between two bounding boxes"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        
        center1 = ((x1 + x2) / 2, (y1 + y2) / 2)
        center2 = ((x3 + x4) / 2, (y3 + y4) / 2)
        
        distance = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
        box1_size = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return distance / box1_size  # Normalize by box size 