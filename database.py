import sqlite3
import os
from datetime import datetime
import hashlib
import json
from typing import List, Optional, Tuple, Any

class DatabaseManager:
    """
    Database management system for the CCTV surveillance application.
    
    Member 2's Responsibilities:
    1. Implement efficient database operations
    2. Optimize queries and indexing
    3. Manage data persistence
    4. Implement analytics queries
    """
    
    def __init__(self, db_path: str = "surveillance.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self.initialize_database()

    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            return True
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            return False

    def close(self):
        """Close database connection"""
        if self.conn:
            try:
                self.conn.close()
            except sqlite3.Error as e:
                print(f"Error closing database connection: {e}")

    def initialize_database(self):
        """Initialize database with required tables"""
        if not self.connect():
            return False

        try:
            # Create users table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Create cameras table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS cameras (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                port INTEGER NOT NULL,
                location TEXT,
                status TEXT DEFAULT 'inactive',
                last_active TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Create anomalies table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomalies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                anomaly_type TEXT NOT NULL,
                description TEXT,
                confidence FLOAT,
                image_path TEXT,
                video_path TEXT,
                processed BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (camera_id) REFERENCES cameras(id)
            )
            ''')

            # Create system_settings table
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                setting_key TEXT UNIQUE NOT NULL,
                setting_value TEXT,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Create default admin user if not exists
            self.cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
            if self.cursor.fetchone()[0] == 0:
                self.create_user('admin', 'admin123', 'admin@example.com', 'admin')

            # Create default settings if not exists
            default_settings = [
                ('violence_threshold', '0.75', 'Threshold for violence detection'),
                ('min_violence_frames', '2', 'Minimum frames for violence detection'),
                ('proximity_threshold', '0.8', 'Threshold for proximity detection'),
                ('movement_threshold', '0.15', 'Threshold for movement detection'),
                ('email_notifications', 'true', 'Enable/disable email notifications'),
                ('notification_recipients', '[]', 'List of email recipients for notifications')
            ]
            
            for key, value, desc in default_settings:
                self.cursor.execute('''
                INSERT OR IGNORE INTO system_settings (setting_key, setting_value, description)
                VALUES (?, ?, ?)
                ''', (key, value, desc))

            self.conn.commit()
            return True

        except sqlite3.Error as e:
            print(f"Error initializing database: {e}")
            return False
        finally:
            self.close()

    def create_user(self, username, password, email=None, role='user'):
        """Create a new user"""
        if not self.connect():
            return False

        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            self.cursor.execute('''
            INSERT INTO users (username, password_hash, email, role)
            VALUES (?, ?, ?, ?)
            ''', (username, password_hash, email, role))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            print(f"User {username} already exists")
            return False
        except sqlite3.Error as e:
            print(f"Error creating user: {e}")
            return False
        finally:
            self.close()

    def verify_user(self, username, password):
        """Verify user credentials"""
        if not self.connect():
            return None

        try:
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            self.cursor.execute('''
            SELECT id, role FROM users 
            WHERE username = ? AND password_hash = ?
            ''', (username, password_hash))
            result = self.cursor.fetchone()
            return result
        except sqlite3.Error as e:
            print(f"Error verifying user: {e}")
            return None
        finally:
            self.close()

    def add_camera(self, name, ip_address, port, location=None):
        """Add a new camera"""
        if not self.connect():
            return False

        try:
            self.cursor.execute('''
            INSERT INTO cameras (name, ip_address, port, location)
            VALUES (?, ?, ?, ?)
            ''', (name, ip_address, port, location))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            print(f"Camera {name} already exists")
            return False
        except sqlite3.Error as e:
            print(f"Error adding camera: {e}")
            return False
        finally:
            self.close()

    def log_anomaly(self, camera_id, anomaly_type, description, confidence, image_path=None, video_path=None):
        """Log a new anomaly"""
        if not self.connect():
            return False

        try:
            self.cursor.execute('''
            INSERT INTO anomalies (camera_id, anomaly_type, description, confidence, image_path, video_path)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (camera_id, anomaly_type, description, confidence, image_path, video_path))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error logging anomaly: {e}")
            return False
        finally:
            self.close()

    def get_anomalies(self, camera_id=None, start_date=None, end_date=None, limit=100):
        """Retrieve anomalies with optional filters"""
        if not self.connect():
            return []

        try:
            query = '''
            SELECT a.*, c.name as camera_name 
            FROM anomalies a
            LEFT JOIN cameras c ON a.camera_id = c.id
            WHERE 1=1
            '''
            params = []

            if camera_id:
                query += ' AND a.camera_id = ?'
                params.append(camera_id)
            if start_date:
                query += ' AND a.timestamp >= ?'
                params.append(start_date)
            if end_date:
                query += ' AND a.timestamp <= ?'
                params.append(end_date)

            query += ' ORDER BY a.timestamp DESC LIMIT ?'
            params.append(limit)

            self.cursor.execute(query, params)
            results = self.cursor.fetchall()
            return results
        except sqlite3.Error as e:
            print(f"Error retrieving anomalies: {e}")
            return []
        finally:
            self.close()

    def get_setting(self, key):
        """Get a system setting"""
        if not self.connect():
            return None

        try:
            self.cursor.execute('SELECT setting_value FROM system_settings WHERE setting_key = ?', (key,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except sqlite3.Error as e:
            print(f"Error getting setting: {e}")
            return None
        finally:
            self.close()

    def update_setting(self, key, value):
        """Update a system setting"""
        if not self.connect():
            return False

        try:
            self.cursor.execute('''
            UPDATE system_settings 
            SET setting_value = ?, updated_at = CURRENT_TIMESTAMP
            WHERE setting_key = ?
            ''', (value, key))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating setting: {e}")
            return False
        finally:
            self.close()

    def get_camera(self, camera_id):
        """Get camera details"""
        if not self.connect():
            return None

        try:
            self.cursor.execute('SELECT * FROM cameras WHERE id = ?', (camera_id,))
            result = self.cursor.fetchone()
            return result
        except sqlite3.Error as e:
            print(f"Error getting camera: {e}")
            return None
        finally:
            self.close()

    def update_camera_status(self, camera_id, status):
        """Update camera status"""
        if not self.connect():
            return False

        try:
            self.cursor.execute('''
            UPDATE cameras 
            SET status = ?, last_active = CURRENT_TIMESTAMP
            WHERE id = ?
            ''', (status, camera_id))
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            print(f"Error updating camera status: {e}")
            return False
        finally:
            self.close()

    # TODO: Member 2 - Implement analytics methods
    def get_anomaly_statistics(self, time_period: str = 'day') -> dict:
        """Get statistics about anomalies"""
        pass
    
    def get_camera_performance(self, camera_id: int) -> dict:
        """Get performance metrics for a camera"""
        pass
    
    def backup_database(self, backup_path: str):
        """Create a backup of the database"""
        pass 