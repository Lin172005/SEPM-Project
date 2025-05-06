# CCTV Anomaly Detection System

A real-time surveillance system that uses computer vision and machine learning to detect anomalies and potential violent activities in CCTV footage.

## 🚀 Features

- Real-time video processing using YOLO object detection
- Multiple input sources (DroidCam, uploaded videos)
- Advanced anomaly detection algorithms
- Context-aware detection system
- Email notifications for detected anomalies
- Secure user authentication
- Incident logging and management
- Video recording capabilities
- Detailed anomaly logs with filtering options

## 📋 User Stories

### Authentication & User Management
- As a security administrator, I want to log in securely to access the system
- As a user, I want to be able to log out of the system
- As an administrator, I want to manage different user roles and permissions

### Video Processing
- As a security operator, I want to connect to DroidCam for real-time surveillance
- As a user, I want to upload pre-recorded videos for analysis
- As a user, I want to record surveillance footage for later review
- As a user, I want to see real-time video processing with bounding boxes

### Anomaly Detection
- As a security operator, I want to be notified when anomalies are detected
- As a user, I want to receive email notifications for detected incidents
- As a user, I want to see detailed information about detected anomalies
- As a user, I want to filter and search through anomaly logs

### System Management
- As an administrator, I want to configure email notification settings
- As a user, I want to view system statistics and performance metrics
- As a user, I want to manage and organize recorded incidents

## 🎯 Sprint Planning

### Sprint 1: Foundation (2 weeks)
- [x] Project setup and environment configuration
- [x] Basic user authentication system
- [x] YOLO model integration
- [x] Basic video processing pipeline

### Sprint 2: Core Features (2 weeks)
- [x] Real-time video processing
- [x] DroidCam integration
- [x] Basic anomaly detection
- [x] Video recording functionality

### Sprint 3: Advanced Detection (2 weeks)
- [x] Context-aware detection system
- [x] Advanced anomaly detection algorithms
- [x] Email notification system
- [x] Incident logging system

### Sprint 4: User Interface & Management (2 weeks)
- [x] Enhanced user interface
- [x] Anomaly log management
- [x] System statistics and monitoring
- [x] User role management

### Sprint 5: Optimization & Testing (2 weeks)
- [x] Performance optimization
- [x] System testing
- [x] Bug fixes
- [x] Documentation

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- DroidCam app (for mobile camera integration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cctv-anomaly-detection.git
cd cctv-anomaly-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Set up the database:
```bash
python setup_database.py
```

4. Configure email settings:
- Enable 2-Step Verification in your Google Account
- Generate an App Password
- Update the email configuration in the application

### Running the Application

1. Start the application:
```bash
streamlit run app.py
```

2. Access the application at `http://localhost:8501`

## 📦 Project Structure

```
cctv-anomaly-detection/
├── app.py                 # Main application file
├── advanced_detection.py  # Advanced anomaly detection algorithms
├── context_aware_detection.py  # Context-aware detection system
├── database.py           # Database management
├── requirements.txt      # Project dependencies
├── setup_database.py     # Database setup script
├── recordings/          # Directory for recorded videos
├── incidents/          # Directory for incident images
└── anomaly_logs/       # Directory for anomaly logs
```

## 🔧 Configuration

### Email Configuration
1. Go to Google Account settings
2. Enable 2-Step Verification
3. Generate App Password
4. Update `EMAIL_ADDRESS` and `EMAIL_PASSWORD` in the application

### DroidCam Setup
1. Install DroidCam on your mobile device
2. Connect to the same WiFi network
3. Note the IP address and port from the DroidCam app
4. Enter these details in the application

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- YOLO team for the object detection model
- Streamlit for the web application framework
- OpenCV for computer vision capabilities
