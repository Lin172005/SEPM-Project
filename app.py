import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO
from advanced_detection import AdvancedAnomalyDetector
import time
import requests
import hashlib
import urllib.parse
from datetime import datetime
import torch
import threading
from pathlib import Path
import json
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from database import DatabaseManager
from context_aware_detection import ContextAwareDetector

# Initialize database
db = DatabaseManager()

# Login page
def login_page():
    st.title("CCTV Violence Detection System - Login")
    
    if not st.session_state.logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            result = db.verify_user(username, password)
            if result:
                st.session_state.logged_in = True
                st.session_state.user_id = result[0]
                st.session_state.user_role = result[1]
                st.rerun()
            else:
                st.error("Invalid username or password")
        
        return False
    return True

# Initialize YOLO model
@st.cache_resource
def load_model():
    try:
        # Ensure CUDA is available if possible
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Load the model with specific parameters
        model = YOLO('yolov8n.pt')
        model.to(device)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def process_frame(frame, model):
    try:
        # Run YOLO detection
        results = model.predict(frame, conf=0.5, verbose=False)
        
        # Extract bounding boxes for people only (class 0)
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:  # Only process people
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2])
        
        return detections
    except Exception as e:
        print(f"Error in process_frame: {e}")
        return []

def check_droidcam_connection(ip, port):
    try:
        # Try different URL formats
        urls = [
            f"http://{ip}:{port}/video",
            f"http://{ip}:{port}/mjpegfeed",
            f"http://{ip}:{port}/videofeed"
        ]
        
        for url in urls:
            try:
                response = requests.get(url, timeout=3, stream=True)
                if response.status_code == 200:
                    return True, url
            except:
                continue
        return False, None
    except:
        return False, None

def create_capture_device(ip, port):
    """Try different methods to create a video capture device"""
    urls = [
        f"http://{ip}:{port}/video",
        f"http://{ip}:{port}/mjpegfeed",
        f"http://{ip}:{port}/videofeed"
    ]
    
    # First try FFMPEG backend
    for url in urls:
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if cap.isOpened():
                return cap, url
        except:
            continue
    
    # Then try default backend
    for url in urls:
        try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                return cap, url
        except:
            continue
    
    return None, None

def create_droidcam_url(ip, port):
    # Clean and format the IP address
    ip = ip.strip()
    # Ensure the URL is properly formatted
    url = f"http://{ip}:{port}/video"
    return urllib.parse.quote(url, safe=':/')

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ['recordings', 'incidents']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        st.write(f"✅ Checked directory: {directory}")

class VideoWriter:
    def __init__(self, save_path, fps, frame_size):
        self.save_path = save_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.is_recording = False
        # Ensure the save directory exists
        os.makedirs(self.save_path, exist_ok=True)

    def start(self):
        if not self.is_recording:
            try:
                filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                filepath = os.path.join(self.save_path, filename)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(filepath, fourcc, self.fps, self.frame_size)
                self.is_recording = True
                return filepath
            except Exception as e:
                st.error(f"Error starting recording: {str(e)}")
                return None

    def write(self, frame):
        if self.is_recording and self.writer is not None:
            try:
                self.writer.write(frame)
            except Exception as e:
                st.error(f"Error writing frame: {str(e)}")

    def stop(self):
        if self.is_recording and self.writer is not None:
            try:
                self.writer.release()
            except Exception as e:
                st.error(f"Error stopping recording: {str(e)}")
            finally:
                self.is_recording = False
                self.writer = None

def save_incident(frame, timestamp):
    """Save incident frame with error handling"""
    try:
        # Ensure incidents directory exists
        os.makedirs('incidents', exist_ok=True)
        
        # Create filename with timestamp
        filename = f"incident_{timestamp}.jpg"
        filepath = os.path.join('incidents', filename)
        
        # Save the frame
        cv2.imwrite(filepath, frame)
        return filepath
    except Exception as e:
        st.error(f"Error saving incident: {str(e)}")
        return None

# Email configuration
EMAIL_ADDRESS = "praiselin172@gmail.com"  # Your Gmail address
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_PASSWORD = ""  # Your Gmail App Password

def test_email_connection():
    """Test the email configuration"""
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            return True, "Email configuration is working correctly!"
    except smtplib.SMTPAuthenticationError:
        return False, "Authentication failed. Please check your App Password."
    except Exception as e:
        return False, f"Connection error: {str(e)}"

def send_anomaly_email(anomalies, frame, timestamp):
    """Send email notification for multiple anomalies"""
    try:
        if not EMAIL_PASSWORD:
            st.error("Please set up your Gmail App Password in the sidebar")
            return
            
        # Test connection first
        success, message = test_email_connection()
        if not success:
            st.error(f"Email test failed: {message}")
            return
            
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_ADDRESS
        msg['To'] = EMAIL_ADDRESS
        msg['Subject'] = f"Multiple Anomalies Detected - {timestamp}"
        
        # Create email body
        body = f"""
        Multiple anomalies have been detected in the surveillance system:
        
        Time: {timestamp}
        Number of anomalies: {len(anomalies)}
        
        Detected anomalies:
        """
        for anomaly_type, description in anomalies:
            body += f"- {anomaly_type}: {description}\n"
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach the frame as an image
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        image = MIMEImage(img_bytes, name='anomaly_frame.jpg')
        msg.attach(image)
        
        # Send email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
            
        st.success("Email notification sent successfully!")
    except smtplib.SMTPAuthenticationError:
        st.error("""
        Gmail authentication failed. Please:
        1. Make sure you're using an App Password, not your regular Gmail password
        2. Check if 2-Step Verification is enabled
        3. Verify the App Password is correct
        """)
    except Exception as e:
        st.error(f"Failed to send email notification: {str(e)}")

def process_video_stream(cap, model, anomaly_detector):
    # Initialize context-aware detector
    context_detector = ContextAwareDetector()
    
    # Ensure directories exist before starting
    ensure_directories()
    os.makedirs("anomaly_logs", exist_ok=True)
    
    # Create interface layout with custom CSS
    st.markdown("""
        <style>
        .main-content { max-width: 1200px; margin: auto; }
        .stVideo { width: 100%; height: auto; }
        .status-box {
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 10px;
        }
        .alert { background-color: rgba(255, 0, 0, 0.1); border: 1px solid red; }
        .normal { background-color: rgba(0, 255, 0, 0.1); border: 1px solid green; }
        </style>
    """, unsafe_allow_html=True)

    # Create columns with custom widths
    col1, col2 = st.columns([7, 3])
    
    with col1:
        st.markdown("### Live Surveillance Feed")
        video_placeholder = st.empty()
        status_text = st.empty()
    
    with col2:
        st.markdown("### Control Panel")
        
        # Initialize recording state if not exists
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
            
        # Initialize record button key if not exists
        if 'record_button_key' not in st.session_state:
            st.session_state.record_button_key = f"record_btn_{time.time()}_{np.random.randint(0, 1000000)}"
        
        # Create record button with persistent key
        record_status = st.button(
            "🔴 Stop Recording" if st.session_state.is_recording else "⚫ Start Recording",
            key=st.session_state.record_button_key
        )
        
        st.markdown("### System Status")
        monitoring_status = st.empty()
        
        st.markdown("### Recent Incidents")
        incident_text = st.empty()
        
        st.markdown("### System Statistics")
        stats_text = st.empty()
    
    # Initialize variables
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 1.0
    last_fps_update = time.time()
    current_fps = 0
    skip_frames = 2
    frame_skip_counter = 0
    recent_incidents = []
    
    # Initialize video writer with error handling
    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = VideoWriter("recordings", 30, (frame_width, frame_height))
    except Exception as e:
        st.error(f"Error initializing video writer: {str(e)}")
        return
    
    try:
        while True:
            try:
                # Handle recording control
                if record_status:
                    st.session_state.is_recording = not st.session_state.is_recording
                    if st.session_state.is_recording:
                        filepath = video_writer.start()
                        if filepath:
                            st.success(f"📹 Recording started: {filepath}")
                    else:
                        video_writer.stop()
                        st.success("✅ Recording saved")
                
                # Read frame
                ret, frame = cap.read()
                if not ret or frame is None:
                    status_text.error("📡 Camera connection lost. Attempting to reconnect...")
                    time.sleep(1)
                    continue
                
                # Frame skipping for performance
                frame_skip_counter = (frame_skip_counter + 1) % skip_frames
                if frame_skip_counter != 0:
                    continue
                
                # Save frame if recording
                if st.session_state.is_recording:
                    video_writer.write(frame)
                
                # Resize frame for better performance
                display_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_NEAREST)
                
                # Process frame with YOLO
                try:
                    results = model(display_frame, conf=0.4, verbose=False)[0]
                    
                    # Extract bounding boxes for people
                    detections = []
                    for box in results.boxes:
                        cls = int(box.cls[0])
                        if cls == 0:  # Only process people
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            detections.append([x1, y1, x2, y2])
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Detect anomalies using context-aware approach
                    context_anomalies = context_detector.detect_contextual_anomalies(display_frame, detections)
                    
                    # Combine with traditional anomaly detection
                    traditional_anomalies = anomaly_detector.detect_anomalies(display_frame, detections)
                    
                    # Merge anomalies
                    all_anomalies = context_anomalies + traditional_anomalies
                    
                    # Handle detected anomalies
                    if all_anomalies:
                        for anomaly_type, description in all_anomalies:
                            # Generate timestamps
                            file_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                            log_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            
                            # Save incident frame and log
                            img_path = os.path.join('incidents', f'anomaly_{file_timestamp}.jpg')
                            cv2.imwrite(img_path, display_frame)
                            
                            # Save to log with formatted timestamp
                            save_anomaly_log(anomaly_type, description, img_path, log_timestamp)
                            
                            # Update recent incidents
                            recent_incidents.insert(0, f"🚨 {file_timestamp}: {description}")
                            if len(recent_incidents) > 5:
                                recent_incidents.pop()
                    
                    # Add timestamp and recording indicator
                    display_timestamp = datetime.now().strftime('%H:%M:%S')
                    cv2.putText(display_frame, display_timestamp, 
                              (10, display_frame.shape[0] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if st.session_state.is_recording:
                        cv2.circle(display_frame, (20, 20), 8, (0, 0, 255), -1)
                    
                    # Convert BGR to RGB for display
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    
                    # Update video display
                    video_placeholder.image(display_frame, channels="RGB", use_column_width=True)
                    
                    # Update FPS calculation
                    frame_count += 1
                    current_time = time.time()
                    if current_time - last_fps_update >= fps_update_interval:
                        current_fps = frame_count / (current_time - start_time)
                        last_fps_update = current_time
                    
                    # Update interface elements less frequently
                    if frame_count % 10 == 0:
                        # Update monitoring status
                        status_class = "alert" if all_anomalies else "normal"
                        monitoring_status.markdown(f"""
                        <div class="status-box {status_class}">
                        🎥 <b>Camera</b>: {'🔴 Recording' if st.session_state.is_recording else '🟢 Active'}<br>
                        👥 <b>People</b>: {len(detections)}<br>
                        ⚠️ <b>Status</b>: {'🚨 ALERT' if all_anomalies else '✅ Normal'}<br>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Update incidents
                        if recent_incidents:
                            incident_text.markdown("<br>".join([
                                f"<div class='status-box alert'>{incident}</div>" 
                                for incident in recent_incidents[:3]
                            ]), unsafe_allow_html=True)
                        
                        # Update stats
                        stats_text.markdown(f"""
                        <div class="status-box">
                        ⏱️ <b>FPS</b>: {current_fps:.1f}<br>
                        🕒 <b>Runtime</b>: {int(current_time - start_time)}s<br>
                        💾 <b>Incidents</b>: {len(recent_incidents)}
                        </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error in frame processing: {str(e)}")
                    continue
                
            except Exception as e:
                status_text.error(f"Stream error: {str(e)}")
                time.sleep(1)
                continue
                
    except Exception as e:
        st.error(f"System error: {str(e)}")
    finally:
        video_writer.stop()
        if cap is not None:
            cap.release()

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'main'

def save_anomaly_log(anomaly_type, description, image_path, timestamp):
    """Save anomaly information to the database"""
    try:
        # Get the current camera ID (you might want to pass this as a parameter)
        camera_id = 1  # Default camera ID, should be replaced with actual camera ID
        
        # Extract confidence from description if available
        confidence = 0.0
        if '(' in description and ')' in description:
            try:
                confidence = float(description.split('(')[1].split(')')[0])
            except:
                pass
        
        # Log to database
        db.log_anomaly(
            camera_id=camera_id,
            anomaly_type=anomaly_type,
            description=description,
            confidence=confidence,
            image_path=image_path
        )
        
        return True
    except Exception as e:
        st.error(f"Error saving anomaly to database: {str(e)}")
        return False

def save_video_anomaly(frame, anomaly_type, description, frame_number):
    """Save anomaly information for uploaded video"""
    log_dir = "anomaly_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "anomaly_log.json")
    
    # Generate current timestamp
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Load existing logs
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    else:
        logs = []
    
    # Save screenshot
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    img_path = os.path.join('incidents', f'video_anomaly_{timestamp_str}.jpg')
    cv2.imwrite(img_path, frame)
    
    # Add log entry
    log_entry = {
        'timestamp': current_time,
        'type': anomaly_type,
        'description': description,
        'image_path': img_path,
        'source': 'uploaded_video',
        'frame_number': frame_number
    }
    
    logs.append(log_entry)
    
    # Save updated logs
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=4)
    
    return img_path

def show_anomaly_logs():
    """Display the anomaly logs page"""
    st.title("Anomaly Detection Logs")
    
    # Add filters
    st.sidebar.title("Filters")
    
    # Get all cameras for filter
    try:
        db.connect()
        db.cursor.execute("SELECT id, name FROM cameras")
        cameras = db.cursor.fetchall()
        
        # Add default camera if none exists
        if not cameras:
            db.cursor.execute("""
            INSERT INTO cameras (name, ip_address, port, location)
            VALUES ('Default Camera', '0.0.0.0', 0, 'Default Location')
            """)
            db.conn.commit()
            db.cursor.execute("SELECT id, name FROM cameras")
            cameras = db.cursor.fetchall()
    except Exception as e:
        st.error(f"Database error: {str(e)}")
        cameras = []
    finally:
        db.close()
    
    camera_options = ['All'] + [f"{name} (ID: {id})" for id, name in cameras]
    selected_camera = st.sidebar.selectbox("Camera", camera_options)
    
    # Get anomalies from JSON file first to determine date range
    json_anomalies = []
    log_file = os.path.join('anomaly_logs', 'anomaly_log.json')
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                json_anomalies = json.load(f)
                st.sidebar.info(f"Found {len(json_anomalies)} anomalies in JSON file")
                
                # Find min and max dates in anomalies
                dates = []
                for anomaly in json_anomalies:
                    if 'timestamp' in anomaly:
                        try:
                            timestamp = anomaly['timestamp']
                            date = datetime.strptime(timestamp.split()[0], '%Y-%m-%d').date()
                            dates.append(date)
                        except:
                            continue
                
                if dates:
                    min_date = min(dates)
                    max_date = max(dates)
                    st.sidebar.info(f"Available date range: {min_date} to {max_date}")
                    
                    # Date range filter with default values
                    col1, col2 = st.sidebar.columns(2)
                    with col1:
                        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
                    with col2:
                        end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
                else:
                    st.warning("No valid dates found in anomalies")
                    start_date = end_date = None
                    
        except json.JSONDecodeError:
            st.error("Error reading anomaly log file. The file may be corrupted.")
            return
    else:
        st.warning("Anomaly log file not found")
        start_date = end_date = None
    
    # Get anomalies from database
    camera_id = None
    if selected_camera != 'All':
        try:
            camera_id = int(selected_camera.split('ID: ')[1].split(')')[0])
        except:
            st.warning("Invalid camera ID format")
    
    try:
        db.connect()
        db_anomalies = db.get_anomalies(
            camera_id=camera_id,
            start_date=start_date.strftime('%Y-%m-%d') if start_date else None,
            end_date=end_date.strftime('%Y-%m-%d') if end_date else None
        )
    except Exception as e:
        st.error(f"Error getting anomalies from database: {str(e)}")
        db_anomalies = []
    finally:
        db.close()
    
    # Filter JSON anomalies by date if specified
    if start_date or end_date:
        filtered_json_anomalies = []
        for anomaly in json_anomalies:
            if 'timestamp' in anomaly:
                try:
                    # Try different timestamp formats
                    timestamp = anomaly['timestamp']
                    try:
                        anomaly_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').date()
                    except ValueError:
                        try:
                            anomaly_date = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S.%f').date()
                        except ValueError:
                            try:
                                anomaly_date = datetime.strptime(timestamp, '%Y-%m-%d').date()
                            except ValueError:
                                continue
                    
                    if start_date and anomaly_date < start_date:
                        continue
                    if end_date and anomaly_date > end_date:
                        continue
                    filtered_json_anomalies.append(anomaly)
                except Exception as e:
                    st.warning(f"Error processing timestamp for anomaly: {str(e)}")
                    continue
        json_anomalies = filtered_json_anomalies
        st.sidebar.info(f"After date filtering: {len(json_anomalies)} anomalies")
    
    # Combine and sort anomalies
    all_anomalies = []
    
    # Add database anomalies
    for anomaly in db_anomalies:
        all_anomalies.append({
            'type': anomaly[3],
            'timestamp': anomaly[2],
            'description': anomaly[4],
            'confidence': anomaly[5],
            'image_path': anomaly[6],
            'source': 'database'
        })
    
    # Add JSON anomalies
    for anomaly in json_anomalies:
        if 'timestamp' in anomaly:
            try:
                confidence = 0.0
                if 'description' in anomaly and '(' in anomaly['description'] and ')' in anomaly['description']:
                    try:
                        confidence = float(anomaly['description'].split('(')[1].split(')')[0])
                    except (ValueError, IndexError):
                        pass
                
                all_anomalies.append({
                    'type': anomaly.get('type', 'Unknown'),
                    'timestamp': anomaly['timestamp'],
                    'description': anomaly.get('description', 'No description'),
                    'confidence': confidence,
                    'image_path': anomaly.get('image_path', ''),
                    'source': 'json'
                })
            except Exception as e:
                st.warning(f"Error processing JSON anomaly: {str(e)}")
                continue
    
    # Sort by timestamp
    try:
        all_anomalies.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S') if isinstance(x['timestamp'], str) else x['timestamp'], reverse=True)
    except Exception as e:
        st.warning(f"Error sorting anomalies: {str(e)}")
    
    if not all_anomalies:
        st.info("No anomaly logs found.")
        st.info("This could be because:")
        st.info("1. No anomalies have been detected yet")
        st.info("2. The date filters are excluding all anomalies")
        st.info("3. There might be an issue with the database connection")
        return
    
    # Display logs
    st.write(f"Found {len(all_anomalies)} anomalies")
    
    # Create columns for layout
    for anomaly in all_anomalies:
        with st.container():
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if os.path.exists(anomaly['image_path']):
                    img = cv2.imread(anomaly['image_path'])
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img, caption="Anomaly Image", use_column_width=True)
                else:
                    st.warning(f"Image not found: {anomaly['image_path']}")
            
            with col2:
                st.markdown(f"**Type:** {anomaly['type']}")
                st.markdown(f"**Time:** {anomaly['timestamp']}")
                st.markdown(f"**Description:** {anomaly['description']}")
                st.markdown(f"**Confidence:** {anomaly['confidence']:.2f}")
                st.markdown(f"**Source:** {anomaly['source']}")
            
            st.markdown("---")

def process_uploaded_video(uploaded_file, model, anomaly_detector):
    """Process uploaded video file"""
    # Create necessary directories
    os.makedirs('incidents', exist_ok=True)
    os.makedirs('anomaly_logs', exist_ok=True)
    
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create placeholders
    video_placeholder = st.empty()
    progress_bar = st.progress(0)
    anomaly_text = st.empty()
    stats_text = st.empty()
    
    frame_count = 0
    start_time = time.time()
    anomalies_detected = 0
    
    # Initialize context-aware detector
    context_detector = ContextAwareDetector()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with YOLO
            try:
                results = model(frame, conf=0.4, verbose=False)[0]
                
                # Extract bounding boxes for people
                detections = []
                for box in results.boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # Only process people
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        detections.append([x1, y1, x2, y2])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Detect anomalies using context-aware approach
                context_anomalies = context_detector.detect_contextual_anomalies(frame, detections)
                
                # Combine with traditional anomaly detection
                traditional_anomalies = anomaly_detector.detect_anomalies(frame, detections)
                
                # Merge anomalies
                all_anomalies = context_anomalies + traditional_anomalies
                
                # Handle detected anomalies
                if all_anomalies:
                    for anomaly_type, description in all_anomalies:
                        # Save anomaly with current timestamp
                        save_video_anomaly(frame, anomaly_type, description, frame_count)
                        anomalies_detected += 1
                        anomaly_text.text(f"⚠️ {anomaly_type}: {description}")
                else:
                    anomaly_text.text("✅ No anomalies detected")
                
            except Exception as e:
                st.error(f"Error processing frame {frame_count}: {str(e)}")
                continue
            
            # Convert BGR to RGB
            display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(display_frame, channels="RGB")
            
            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            
            # Update stats
            elapsed_time = time.time() - start_time
            current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            stats_text.markdown(f"""
            **Processing Statistics:**
            - Frames Processed: {frame_count}/{total_frames}
            - FPS: {current_fps:.2f}
            - Anomalies Detected: {anomalies_detected}
            """)
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
    finally:
        cap.release()
        os.unlink(video_path)
        
    if anomalies_detected > 0:
        st.success(f"Video processing completed! Found {anomalies_detected} anomalies. Check the Anomaly Logs page to view them.")
    else:
        st.info("Video processing completed. No anomalies were detected.")

def main():
    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'record_button_key' not in st.session_state:
        st.session_state.record_button_key = f"record_btn_{time.time()}_{np.random.randint(0, 1000000)}"
    
    # Check login status
    if not login_page():
        return
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Main", "Anomaly Logs"])
    
    if page == "Anomaly Logs":
        show_anomaly_logs()
        return
    
    st.title("CCTV Violence Detection System")
    
    # Add logout button
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.user_role = None
        st.rerun()
    
    # Initialize models
    model = load_model()
    if model is None:
        st.error("Failed to load YOLO model. Please check your installation.")
        return
    
    anomaly_detector = AdvancedAnomalyDetector()
    
    # Sidebar for input selection
    input_type = st.sidebar.radio(
        "Select Input Source",
        ["Upload Video", "DroidCam"]
    )
    
    if input_type == "Upload Video":
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None:
            process_uploaded_video(uploaded_file, model, anomaly_detector)
    
    else:  # DroidCam
        st.info("📹 DroidCam CCTV Surveillance Mode")
        st.markdown("""
        ### Setup Instructions:
        1. Install DroidCam on your phone
        2. Connect phone to the same WiFi network
        3. Enter the IP and port shown in DroidCam app
        4. Start surveillance
        """)
        
        # DroidCam connection settings
        col1, col2 = st.columns(2)
        with col1:
            droidcam_ip = st.text_input("DroidCam IP Address", "10.3.111.46")
        with col2:
            droidcam_port = st.number_input("DroidCam Port", 4747)
        
        # Start surveillance
        if st.button("Start Surveillance"):
            st.info("🎥 Initializing CCTV surveillance...")
            
            # Try to create capture device
            cap, url = create_capture_device(droidcam_ip, droidcam_port)
            
            if cap is None:
                st.error("❌ Failed to connect to camera. Please check:")
                st.markdown("""
                1. Is DroidCam app running?
                2. Is the IP address correct?
                3. Can you access the stream in a web browser?
                4. Are both devices on the same network?
                """)
                return
            
            try:
                # Configure camera settings
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                st.success(f"🚀 Connected to camera at {url}")
                process_video_stream(cap, model, anomaly_detector)
            except Exception as e:
                st.error(f"⚠️ Surveillance error: {str(e)}")
            finally:
                if cap is not None:
                    cap.release()
                st.info("📴 Surveillance system stopped")

    # Add email configuration section
    st.sidebar.header("Gmail Configuration")
    st.sidebar.info("""
    To set up Gmail:
    1. Go to https://myaccount.google.com/
    2. Click on 'Security'
    3. Enable '2-Step Verification' if not already enabled
    4. Go back to Security page
    5. Under '2-Step Verification', click on 'App passwords'
    6. Select 'Mail' and 'Other' device
    7. Name it 'CCTV Anomaly Detection'
    8. Copy the 16-character password
    9. Paste it below
    """)
    
    email_address = st.sidebar.text_input("Gmail Address", EMAIL_ADDRESS)
    email_password = st.sidebar.text_input("Gmail App Password", type="password")
    
    if email_password:
        global EMAIL_PASSWORD
        EMAIL_PASSWORD = email_password
        st.sidebar.success("Gmail settings updated!")
        
        # Add test email button
        if st.sidebar.button("Test Email Configuration"):
            success, message = test_email_connection()
            if success:
                st.sidebar.success(message)
            else:
                st.sidebar.error(message)

if __name__ == "__main__":
    main() 