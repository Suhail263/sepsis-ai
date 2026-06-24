"""
Early Sepsis Risk Assessment and Dual Treatment Recommendation System
Using Allopathy and Siddha - Main Flask Application
"""
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
import os, json, secrets
from prediction import predict_sepsis

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# ── Database config (SQLite default - no setup required!) ──────────────────────
DB_USER = os.environ.get('DB_USER', '')
if DB_USER:
    DB_PASS = os.environ.get('DB_PASS', '')
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_NAME = os.environ.get('DB_NAME', 'sepsis_db')
    app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}'
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sepsis.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# ── Models ─────────────────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='doctor')
    full_name = db.Column(db.String(200))
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Patient(db.Model):
    __tablename__ = 'patients'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    patient_name = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    temperature = db.Column(db.Float)
    heart_rate = db.Column(db.Integer)
    respiratory_rate = db.Column(db.Integer)
    systolic_bp = db.Column(db.Integer)
    diastolic_bp = db.Column(db.Integer)
    spo2 = db.Column(db.Float)
    blood_sugar = db.Column(db.Float)
    wbc_count = db.Column(db.Float)
    platelet_count = db.Column(db.Float)
    lactate_level = db.Column(db.Float)
    creatinine_level = db.Column(db.Float)
    urine_output = db.Column(db.Float)
    mental_status = db.Column(db.String(100))
    infection_source = db.Column(db.String(200))
    symptoms = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='patient', lazy=True)

class Prediction(db.Model):
    __tablename__ = 'predictions'
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey('patients.id'))
    risk_level = db.Column(db.String(50))
    risk_percentage = db.Column(db.Float)
    confidence_score = db.Column(db.Float)
    severity_score = db.Column(db.Float)
    feature_importance = db.Column(db.Text)
    allopathy_rec = db.Column(db.Text)
    siddha_rec = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ActivityLog(db.Model):
    __tablename__ = 'activity_logs'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    action = db.Column(db.String(255))
    details = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# ── Helpers ────────────────────────────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def log_activity(user_id, action, details=''):
    try:
        log = ActivityLog(user_id=user_id, action=action, details=details)
        db.session.add(log)
        db.session.commit()
    except:
        pass

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username  = request.form.get('username', '').strip()
        email     = request.form.get('email', '').strip()
        password  = request.form.get('password', '')
        full_name = request.form.get('full_name', '').strip()
        role      = request.form.get('role', 'doctor')

        if User.query.filter_by(username=username).first():
            flash('Username already exists.', 'danger')
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash('Email already registered.', 'danger')
            return render_template('register.html')

        user = User(username=username, email=email,
                    password_hash=generate_password_hash(password),
                    full_name=full_name, role=role)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            session['user_id']   = user.id
            session['username']  = user.username
            session['role']      = user.role
            session['full_name'] = user.full_name or user.username
            log_activity(user.id, 'LOGIN', f'User {username} logged in')
            flash(f'Welcome back, {user.full_name or username}!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    if 'user_id' in session:
        log_activity(session['user_id'], 'LOGOUT')
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    total_patients = Patient.query.filter_by(user_id=session['user_id']).count()
    total_preds    = db.session.query(Prediction).join(Patient).filter(
        Patient.user_id == session['user_id']).count()
    high_risk = db.session.query(Prediction).join(Patient).filter(
        Patient.user_id == session['user_id'],
        Prediction.risk_level.in_(['High Risk', 'Critical Risk'])).count()
    recent = db.session.query(Patient, Prediction).join(Prediction).filter(
        Patient.user_id == session['user_id']).order_by(
        Prediction.created_at.desc()).limit(10).all()

    # Chart data
    risk_counts = {}
    for rl in ['No Sepsis', 'Mild Risk', 'Moderate Risk', 'High Risk', 'Critical Risk']:
        c = db.session.query(Prediction).join(Patient).filter(
            Patient.user_id == session['user_id'],
            Prediction.risk_level == rl).count()
        risk_counts[rl] = c

    # Monthly trend (last 6 months)
    monthly = []
    for i in range(5, -1, -1):
        d = datetime.utcnow() - timedelta(days=30 * i)
        c = db.session.query(Prediction).join(Patient).filter(
            Patient.user_id == session['user_id'],
            db.extract('month', Prediction.created_at) == d.month,
            db.extract('year', Prediction.created_at) == d.year).count()
        monthly.append({'month': d.strftime('%b %Y'), 'count': c})

    return render_template('dashboard.html',
        total_patients=total_patients, total_preds=total_preds,
        high_risk=high_risk, recent=recent,
        risk_counts=json.dumps(risk_counts), monthly=json.dumps(monthly))

@app.route('/assess', methods=['GET', 'POST'])
@login_required
def assess():
    if request.method == 'POST':
        symptoms = request.form.getlist('symptoms')
        form_data = dict(request.form)
        form_data['symptoms'] = symptoms

        # Save patient
        patient = Patient(
            user_id=session['user_id'],
            patient_name=request.form.get('patient_name', 'Unknown'),
            age=int(request.form.get('age', 30)),
            gender=request.form.get('gender', 'Male'),
            temperature=float(request.form.get('temperature', 37)),
            heart_rate=int(request.form.get('heart_rate', 80)),
            respiratory_rate=int(request.form.get('respiratory_rate', 16)),
            systolic_bp=int(request.form.get('systolic_bp', 120)),
            diastolic_bp=int(request.form.get('diastolic_bp', 80)),
            spo2=float(request.form.get('spo2', 98)),
            blood_sugar=float(request.form.get('blood_sugar', 100)),
            wbc_count=float(request.form.get('wbc_count', 8)),
            platelet_count=float(request.form.get('platelet_count', 250)),
            lactate_level=float(request.form.get('lactate_level', 1.0)),
            creatinine_level=float(request.form.get('creatinine_level', 0.9)),
            urine_output=float(request.form.get('urine_output', 60)),
            mental_status=request.form.get('mental_status', 'Alert'),
            infection_source=request.form.get('infection_source', 'Unknown'),
            symptoms=','.join(symptoms)
        )
        db.session.add(patient)
        db.session.flush()

        # Predict
        result = predict_sepsis(form_data)

        # Save prediction
        pred = Prediction(
            patient_id=patient.id,
            risk_level=result['risk_level'],
            risk_percentage=result['risk_percentage'],
            confidence_score=result['confidence_score'],
            severity_score=result['severity_score'],
            feature_importance=json.dumps(result['feature_importance']),
            allopathy_rec=json.dumps(result['allopathy']),
            siddha_rec=json.dumps(result['siddha'])
        )
        db.session.add(pred)
        db.session.commit()
        log_activity(session['user_id'], 'PREDICTION', f'Patient: {patient.patient_name}, Risk: {result["risk_level"]}')

        return render_template('result.html', result=result, patient=patient,
                               prediction=pred,
                               fi_json=json.dumps(result['feature_importance']),
                               prob_json=json.dumps(result['probability_distribution']))
    return render_template('assess.html')

@app.route('/patients')
@login_required
def patients():
    page = request.args.get('page', 1, type=int)
    pats = Patient.query.filter_by(user_id=session['user_id']).order_by(
        Patient.created_at.desc()).paginate(page=page, per_page=15)
    return render_template('patients.html', patients=pats)

@app.route('/patient/<int:pid>')
@login_required
def patient_detail(pid):
    patient = Patient.query.filter_by(id=pid, user_id=session['user_id']).first_or_404()
    preds   = Prediction.query.filter_by(patient_id=pid).order_by(Prediction.created_at.desc()).all()
    return render_template('patient_detail.html', patient=patient, predictions=preds)

@app.route('/api/stats')
@login_required
def api_stats():
    uid = session['user_id']
    stats = {
        'total_patients': Patient.query.filter_by(user_id=uid).count(),
        'total_predictions': db.session.query(Prediction).join(Patient).filter(Patient.user_id == uid).count(),
        'high_risk': db.session.query(Prediction).join(Patient).filter(
            Patient.user_id == uid,
            Prediction.risk_level.in_(['High Risk', 'Critical Risk'])).count(),
    }
    return jsonify(stats)

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user = User.query.get(session['user_id'])
    if request.method == 'POST':
        user.full_name = request.form.get('full_name', user.full_name)
        new_pass = request.form.get('new_password', '')
        if new_pass:
            user.password_hash = generate_password_hash(new_pass)
        db.session.commit()
        session['full_name'] = user.full_name
        flash('Profile updated successfully!', 'success')
    return render_template('profile.html', user=user)

# ── Init ───────────────────────────────────────────────────────────────────────
@app.cli.command('init-db')
def init_db():
    db.create_all()
    # Create default admin
    if not User.query.filter_by(username='admin').first():
        admin = User(username='admin', email='admin@sepsis.ai',
                     password_hash=generate_password_hash('Admin@123'),
                     role='admin', full_name='System Administrator')
        db.session.add(admin)
        db.session.commit()
        print('Default admin created: admin / Admin@123')
    print('Database initialized!')
    # Create tables when app starts
with app.app_context():
 db.create_all()
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            admin = User(username='admin', email='admin@sepsis.ai',
                         password_hash=generate_password_hash('Admin@123'),
                         role='admin', full_name='System Administrator')
            db.session.add(admin)
            db.session.commit()
    app.run(debug=True, host='0.0.0.0', port=5000)


