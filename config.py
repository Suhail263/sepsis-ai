"""
SepsisAI - Configuration
Supports SQLite (default) and MySQL
"""
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'sepsis-ai-secret-key-change-in-production')
    
    # Auto-detect: use SQLite if no MySQL env vars set
    DB_USER = os.environ.get('DB_USER', '')
    DB_PASS = os.environ.get('DB_PASS', '')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_NAME = os.environ.get('DB_NAME', 'sepsis_db')
    
    if DB_USER:
        SQLALCHEMY_DATABASE_URI = f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}'
    else:
        # Default: SQLite (no setup required!)
        SQLALCHEMY_DATABASE_URI = 'sqlite:///sepsis.db'
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
