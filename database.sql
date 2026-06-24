-- ================================================
-- Sepsis Risk Assessment System - Database Schema
-- ================================================

CREATE DATABASE IF NOT EXISTS sepsis_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE sepsis_db;

-- Users Table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role ENUM('admin','doctor','researcher') DEFAULT 'doctor',
    full_name VARCHAR(200),
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE,
    verification_token VARCHAR(255),
    reset_token VARCHAR(255),
    reset_token_expiry DATETIME,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_login DATETIME
);

-- Patients Table
CREATE TABLE IF NOT EXISTS patients (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    patient_name VARCHAR(200) NOT NULL,
    age INT NOT NULL,
    gender ENUM('Male','Female','Other') NOT NULL,
    temperature FLOAT,
    heart_rate INT,
    respiratory_rate INT,
    systolic_bp INT,
    diastolic_bp INT,
    spo2 FLOAT,
    blood_sugar FLOAT,
    wbc_count FLOAT,
    platelet_count FLOAT,
    lactate_level FLOAT,
    creatinine_level FLOAT,
    urine_output FLOAT,
    mental_status VARCHAR(100),
    infection_source VARCHAR(200),
    symptoms TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Predictions Table
CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    risk_level VARCHAR(50),
    risk_percentage FLOAT,
    confidence_score FLOAT,
    severity_score FLOAT,
    model_used VARCHAR(100),
    feature_importance TEXT,
    shap_values TEXT,
    prediction_time FLOAT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
);

-- Treatment Recommendations Table
CREATE TABLE IF NOT EXISTS treatment_recommendations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    prediction_id INT,
    allopathy_recommendation TEXT,
    siddha_recommendation TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);

-- Reports Table
CREATE TABLE IF NOT EXISTS reports (
    id INT AUTO_INCREMENT PRIMARY KEY,
    patient_id INT,
    prediction_id INT,
    report_type VARCHAR(50),
    report_path VARCHAR(500),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE,
    FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
);

-- Activity Logs Table
CREATE TABLE IF NOT EXISTS activity_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT,
    action VARCHAR(255),
    details TEXT,
    ip_address VARCHAR(50),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL
);

-- Default Admin User (password: Admin@123)
INSERT INTO users (username, email, password_hash, role, full_name, is_active, is_verified)
VALUES ('admin', 'admin@sepsis.ai', 'pbkdf2:sha256:260000$placeholder', 'admin', 'System Administrator', TRUE, TRUE);
