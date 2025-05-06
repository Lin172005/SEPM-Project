import streamlit as st
from typing import Optional, Dict, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

class UIManager:
    """
    UI component manager for the CCTV surveillance application.
    
    Member 3's Responsibilities:
    1. Implement and maintain UI components
    2. Handle user interactions
    3. Manage authentication flow
    4. Implement notification system
    """
    
    def __init__(self):
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'logged_in' not in st.session_state:
            st.session_state.logged_in = False
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None
        if 'user_role' not in st.session_state:
            st.session_state.user_role = None
        if 'is_recording' not in st.session_state:
            st.session_state.is_recording = False
    
    def render_login_page(self) -> bool:
        """Render the login page"""
        st.title("CCTV Violence Detection System - Login")
        
        if not st.session_state.logged_in:
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Login"):
                # TODO: Member 3 - Implement secure login
                pass
            
            return False
        return True
    
    def render_sidebar(self):
        """Render the application sidebar"""
        st.sidebar.title("Navigation")
        
        # Navigation
        page = st.sidebar.radio("Go to", ["Main", "Anomaly Logs"])
        
        # Logout button
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.user_role = None
            st.rerun()
        
        return page
    
    def render_email_config(self):
        """Render email configuration section"""
        st.sidebar.header("Gmail Configuration")
        st.sidebar.info("""
        To set up Gmail:
        1. Enable 2-Step Verification
        2. Generate App Password
        3. Enter credentials below
        """)
        
        email = st.sidebar.text_input("Gmail Address")
        password = st.sidebar.text_input("Gmail App Password", type="password")
        
        return email, password
    
    def send_notification(self, email: str, password: str, subject: str,
                         body: str, image_path: Optional[str] = None):
        """Send email notification"""
        try:
            msg = MIMEMultipart()
            msg['From'] = email
            msg['To'] = email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            if image_path:
                with open(image_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', filename='anomaly.jpg')
                    msg.attach(img)
            
            # TODO: Member 3 - Implement secure email sending
            
            return True, "Email sent successfully!"
        except Exception as e:
            return False, f"Error sending email: {str(e)}"
    
    def render_camera_config(self):
        """Render camera configuration section"""
        st.info("📹 DroidCam CCTV Surveillance Mode")
        st.markdown("""
        ### Setup Instructions:
        1. Install DroidCam on your phone
        2. Connect to same WiFi network
        3. Enter IP and port
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            ip = st.text_input("DroidCam IP Address", "192.168.1.100")
        with col2:
            port = st.number_input("DroidCam Port", 4747)
        
        return ip, port
    
    def render_video_upload(self):
        """Render video upload section"""
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        return uploaded_file
    
    def render_processing_stats(self, frame_count: int, total_frames: int,
                              fps: float, anomalies: int):
        """Render processing statistics"""
        st.markdown(f"""
        **Processing Statistics:**
        - Frames: {frame_count}/{total_frames}
        - FPS: {fps:.2f}
        - Anomalies: {anomalies}
        """)
    
    # TODO: Member 3 - Implement additional UI components 