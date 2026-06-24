#!/usr/bin/env python3
"""
SepsisAI - One-click Setup Script
Run: python setup.py
"""
import subprocess, sys, os

def run(cmd, check=True):
    print(f"  > {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout: print(result.stdout)
    if result.returncode != 0 and check:
        print(f"  [WARN] {result.stderr}")
    return result

print("\n" + "="*60)
print("  SepsisAI - Early Sepsis Risk Assessment System")
print("  Setup Script")
print("="*60)

print("\n[1] Installing Python packages...")
run(f"{sys.executable} -m pip install -r requirements.txt -q")

print("\n[2] Training AI Models (this may take 30-60 seconds)...")
run(f"{sys.executable} model_training.py")

print("\n[3] Setup complete!\n")
print("="*60)
print("  HOW TO RUN:")
print("  python app.py")
print()
print("  Then open: http://localhost:5000")
print()
print("  Default Login:")
print("  Username: admin")
print("  Password: Admin@123")
print()
print("  NOTE: MySQL is optional. The app uses SQLite by default.")
print("  To use MySQL, edit DB_* settings in app.py")
print("="*60 + "\n")
