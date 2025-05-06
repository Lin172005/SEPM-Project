# CCTV Violence Detection System

## Project Structure and Team Responsibilities

### Member 1: Video Processing & Detection Core
**Responsibilities:**
- Implementation and optimization of YOLO model integration
- Real-time video processing pipeline
- Person detection and tracking algorithms
- Frame processing optimization
- Context-aware detection system

**Key Files:**
- `advanced_detection.py`
- `context_aware_detection.py`
- Video processing components in `app.py`

**Technical Focus:**
- Computer Vision
- Deep Learning Models
- Performance Optimization
- Real-time Processing

### Member 2: Database & Logging System
**Responsibilities:**
- Database management and optimization
- Anomaly logging system
- Data persistence implementation
- Query optimization
- Analytics and reporting system

**Key Files:**
- `database.py`
- Logging components in `app.py`
- JSON log handling
- Analytics dashboard

**Technical Focus:**
- Database Design
- Data Management
- System Logging
- Analytics Implementation

### Member 3: UI/UX & Camera Integration
**Responsibilities:**
- Streamlit UI development
- Camera integration (DroidCam)
- User authentication system
- Email notification system
- Frontend optimizations

**Key Files:**
- UI components in `app.py`
- Authentication system
- Email notification system
- Camera connection handling

**Technical Focus:**
- Frontend Development
- User Experience
- System Integration
- Notification Systems

## Getting Started

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up the database
4. Configure email settings
5. Run the application: `streamlit run app.py`

## Development Guidelines

1. Each team member should work in their own branch
2. Use meaningful commit messages
3. Document your code
4. Write unit tests for new features
5. Review each other's code before merging

## Communication

- Regular team meetings for progress updates
- Use GitHub issues for task tracking
- Document major decisions and changes

## Dependencies

- Python 3.10+
- OpenCV
- YOLO
- Streamlit
- SQLite
- Other requirements in `requirements.txt`

## Features

- Real-time video stream processing
- Violence detection using YOLO and custom algorithms
- Multiple camera support
- Email notifications for detected anomalies
- User authentication and role-based access
- Anomaly logging and reporting
- Configurable detection thresholds
- Web-based interface using Streamlit

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/anomaly-detection.git
cd anomaly-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the system:
- Copy `config.py` to `config_local.py` and modify settings as needed
- Set up your email credentials for notifications
- Configure camera settings

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Access the web interface at `http://localhost:8501`

3. Login with default credentials:
- Username: admin
- Password: admin123

4. Configure cameras and detection settings

5. Start monitoring

## Database

The system uses SQLite for data storage. The database is automatically created and initialized on first run.

### Database Schema

- `users`: User authentication and roles
- `cameras`: Camera configurations and status
- `anomalies`: Detected anomalies and related data
- `system_settings`: System configuration settings

## Configuration

### Email Setup

1. Enable 2-Step Verification in your Google Account
2. Generate an App Password
3. Update the email configuration in `config_local.py`

### Camera Setup

1. Add cameras through the web interface
2. Configure IP address and port
3. Test connection before starting monitoring

## Security

- All passwords are hashed using SHA-256
- User authentication required for all operations
- Role-based access control
- Secure email notifications

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- YOLO for object detection
- OpenCV for computer vision
- Streamlit for web interface
- All contributors and users 