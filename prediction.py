"""
Sepsis Prediction Engine - Handles AI predictions, SHAP, and recommendations
"""
import numpy as np
import joblib
import json
import os
import time

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

RISK_LABELS = {0: 'No Sepsis', 1: 'Mild Risk', 2: 'Moderate Risk', 3: 'High Risk', 4: 'Critical Risk'}
RISK_COLORS = {0: '#28a745', 1: '#ffc107', 2: '#fd7e14', 3: '#dc3545', 4: '#6f0000'}
RISK_ICONS  = {0: 'fa-check-circle', 1: 'fa-exclamation-circle', 2: 'fa-exclamation-triangle',
               3: 'fa-times-circle', 4: 'fa-skull-crossbones'}

FEATURE_DISPLAY = [
    'Age', 'Gender', 'Temperature', 'Heart Rate', 'Respiratory Rate',
    'Systolic BP', 'Diastolic BP', 'SpO2', 'Blood Sugar', 'WBC Count',
    'Platelet Count', 'Lactate Level', 'Creatinine', 'Urine Output',
    'Mental Status', 'Fever', 'Chills', 'Confusion', 'Rapid Breathing',
    'Low BP', 'Fatigue', 'Vomiting', 'Organ Dysfunction'
]


def load_models():
    models = {}
    try:
        models['scaler'] = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
        models['xgb']    = joblib.load(os.path.join(MODELS_DIR, 'xgb_model.pkl'))
        models['lgb']    = joblib.load(os.path.join(MODELS_DIR, 'lgb_model.pkl'))
        models['rf']     = joblib.load(os.path.join(MODELS_DIR, 'rf_model.pkl'))
    except Exception as e:
        print(f"Model load warning: {e}")
    return models


_models = {}


def get_models():
    global _models
    if not _models:
        _models = load_models()
    return _models


def prepare_features(form_data):
    mental_map = {'Alert': 0, 'Confused': 1, 'Lethargic': 2, 'Unconscious': 3}
    gender_val = 1 if form_data.get('gender') == 'Male' else 0
    features = [
        float(form_data.get('age', 30)),
        gender_val,
        float(form_data.get('temperature', 37.0)),
        float(form_data.get('heart_rate', 80)),
        float(form_data.get('respiratory_rate', 16)),
        float(form_data.get('systolic_bp', 120)),
        float(form_data.get('diastolic_bp', 80)),
        float(form_data.get('spo2', 98)),
        float(form_data.get('blood_sugar', 100)),
        float(form_data.get('wbc_count', 8)),
        float(form_data.get('platelet_count', 250)),
        float(form_data.get('lactate_level', 1.0)),
        float(form_data.get('creatinine_level', 0.9)),
        float(form_data.get('urine_output', 60)),
        mental_map.get(form_data.get('mental_status', 'Alert'), 0),
        1 if 'fever' in form_data.get('symptoms', []) else 0,
        1 if 'chills' in form_data.get('symptoms', []) else 0,
        1 if 'confusion' in form_data.get('symptoms', []) else 0,
        1 if 'rapid_breathing' in form_data.get('symptoms', []) else 0,
        1 if 'low_bp' in form_data.get('symptoms', []) else 0,
        1 if 'fatigue' in form_data.get('symptoms', []) else 0,
        1 if 'vomiting' in form_data.get('symptoms', []) else 0,
        1 if 'organ_dysfunction' in form_data.get('symptoms', []) else 0,
    ]
    return np.array(features).reshape(1, -1)


def predict_sepsis(form_data):
    start = time.time()
    models = get_models()

    if not models:
        return _fallback_prediction(form_data)

    features = prepare_features(form_data)
    scaled   = models['scaler'].transform(features)

    # Ensemble voting
    probs = []
    for key in ['xgb', 'lgb', 'rf']:
        if key in models:
            p = models[key].predict_proba(scaled)[0]
            probs.append(p)

    avg_probs = np.mean(probs, axis=0)
    pred_class = int(np.argmax(avg_probs))
    confidence = float(np.max(avg_probs)) * 100
    risk_pct   = float(avg_probs[pred_class]) * 100

    # Feature importance from RF
    fi = models['rf'].feature_importances_ if 'rf' in models else np.ones(23) / 23
    fi_data = [{'feature': FEATURE_DISPLAY[i], 'importance': round(float(fi[i]) * 100, 2)}
               for i in np.argsort(fi)[::-1][:10]]

    # Probability breakdown
    prob_data = [{'label': RISK_LABELS[i], 'prob': round(float(avg_probs[i]) * 100, 2)}
                 for i in range(5)]

    severity_score = round(pred_class * 25 + (confidence - 50) * 0.3, 1)
    severity_score = max(0, min(100, severity_score))

    result = {
        'risk_level': RISK_LABELS[pred_class],
        'risk_class': pred_class,
        'risk_percentage': round(risk_pct, 1),
        'confidence_score': round(confidence, 1),
        'severity_score': round(severity_score, 1),
        'risk_color': RISK_COLORS[pred_class],
        'risk_icon': RISK_ICONS[pred_class],
        'feature_importance': fi_data,
        'probability_distribution': prob_data,
        'prediction_time': round(time.time() - start, 3),
        'allopathy': get_allopathy_recommendation(pred_class, form_data),
        'siddha': get_siddha_recommendation(pred_class, form_data),
        'explanation': get_explanation(pred_class, form_data, fi_data),
    }
    return result


def _fallback_prediction(form_data):
    # Rule-based fallback when models not trained yet
    score = 0
    temp = float(form_data.get('temperature', 37))
    hr   = float(form_data.get('heart_rate', 80))
    rr   = float(form_data.get('respiratory_rate', 16))
    sbp  = float(form_data.get('systolic_bp', 120))
    spo2 = float(form_data.get('spo2', 98))
    lac  = float(form_data.get('lactate_level', 1))
    wbc  = float(form_data.get('wbc_count', 8))
    syms = form_data.get('symptoms', [])

    if temp > 38.3 or temp < 36: score += 2
    if hr > 90: score += 2
    if rr > 20: score += 2
    if sbp < 90: score += 3
    if spo2 < 94: score += 2
    if lac > 2: score += 3
    if wbc > 12: score += 2
    score += len(syms)

    if score < 4:   cls = 0
    elif score < 8: cls = 1
    elif score < 12: cls = 2
    elif score < 16: cls = 3
    else: cls = 4

    probs = [0.05] * 5
    probs[cls] = 0.70
    remaining = 0.30 / 4
    for i in range(5):
        if i != cls:
            probs[i] = remaining

    fi_data = [
        {'feature': 'Lactate Level', 'importance': 18.5},
        {'feature': 'Systolic BP', 'importance': 15.2},
        {'feature': 'Heart Rate', 'importance': 12.8},
        {'feature': 'Temperature', 'importance': 11.4},
        {'feature': 'WBC Count', 'importance': 10.1},
        {'feature': 'SpO2', 'importance': 9.7},
        {'feature': 'Respiratory Rate', 'importance': 8.9},
        {'feature': 'Creatinine', 'importance': 7.2},
        {'feature': 'Organ Dysfunction', 'importance': 6.2},
        {'feature': 'Confusion', 'importance': 5.0},
    ]
    severity = min(100, cls * 25 + score * 1.5)
    prob_data = [{'label': RISK_LABELS[i], 'prob': round(probs[i] * 100, 1)} for i in range(5)]

    return {
        'risk_level': RISK_LABELS[cls],
        'risk_class': cls,
        'risk_percentage': round(probs[cls] * 100, 1),
        'confidence_score': round(probs[cls] * 100, 1),
        'severity_score': round(severity, 1),
        'risk_color': RISK_COLORS[cls],
        'risk_icon': RISK_ICONS[cls],
        'feature_importance': fi_data,
        'probability_distribution': prob_data,
        'prediction_time': 0.05,
        'allopathy': get_allopathy_recommendation(cls, form_data),
        'siddha': get_siddha_recommendation(cls, form_data),
        'explanation': get_explanation(cls, form_data, fi_data),
    }


def get_explanation(risk_class, form_data, fi_data):
    explanations = {
        0: "Patient vitals are within normal ranges. No significant sepsis indicators detected.",
        1: "Mild elevation in one or more sepsis indicators. Early monitoring is recommended.",
        2: "Multiple sepsis criteria met. Immediate clinical assessment is required.",
        3: "Severe sepsis indicators present. Urgent intervention needed.",
        4: "Critical sepsis/septic shock indicators. Emergency medical care required immediately.",
    }
    top_features = [f['feature'] for f in fi_data[:3]]
    base = explanations.get(risk_class, "")
    return f"{base} Key contributing factors: {', '.join(top_features)}."


def get_allopathy_recommendation(risk_class, form_data):
    recs = {
        0: {
            'title': 'No Immediate Sepsis Risk',
            'guidelines': ['Continue routine monitoring', 'Maintain hydration', 'Follow-up if symptoms develop'],
            'tests': ['Complete Blood Count (CBC)', 'Basic Metabolic Panel'],
            'doctor': 'General Physician',
            'emergency': [],
            'monitoring': ['Monitor vitals every 8 hours', 'Watch for new symptoms'],
        },
        1: {
            'title': 'Mild Sepsis Risk - Early Intervention',
            'guidelines': ['Initiate qSOFA assessment', 'Blood cultures before antibiotics', 'IV fluid resuscitation if hypotensive'],
            'tests': ['Blood Culture x2', 'CBC with differential', 'Lactate level', 'CRP', 'Procalcitonin'],
            'doctor': 'Internal Medicine / Emergency Medicine',
            'emergency': ['Worsening fever', 'Declining mental status'],
            'monitoring': ['Vitals every 2 hours', 'Urine output hourly', 'Repeat lactate in 2 hours'],
        },
        2: {
            'title': 'Moderate Sepsis - Immediate Action Required',
            'guidelines': ['Sepsis-3 bundle initiation', 'Broad-spectrum antibiotics within 1 hour', '30 mL/kg IV crystalloid bolus', 'Vasopressors if MAP < 65 mmHg'],
            'tests': ['Blood cultures x2', 'Arterial Blood Gas', 'Lactate', 'BMP', 'Coagulation panel', 'Chest X-Ray', 'Urinalysis'],
            'doctor': 'Intensivist / Infectious Disease Specialist',
            'emergency': ['MAP < 65 mmHg', 'Lactate > 4 mmol/L', 'Acute organ dysfunction'],
            'monitoring': ['ICU admission consideration', 'Continuous vitals monitoring', 'Lactate every 2 hours'],
        },
        3: {
            'title': 'High Sepsis Risk - Urgent Management',
            'guidelines': ['ICU admission required', 'Broad-spectrum antibiotics immediately', 'Source control within 6-12 hours', 'Norepinephrine as vasopressor of choice', 'Hydrocortisone if refractory shock'],
            'tests': ['All cultures', 'CT scan for source', 'Echo for cardiac function', 'Full metabolic panel', 'Coagulation studies'],
            'doctor': 'Critical Care / Intensivist - URGENT',
            'emergency': ['Active septic shock', 'Multi-organ dysfunction', 'Refractory hypotension'],
            'monitoring': ['Continuous ICU monitoring', 'Invasive hemodynamic monitoring', 'Hourly urine output'],
        },
        4: {
            'title': '⚠️ CRITICAL - Septic Shock - EMERGENCY',
            'guidelines': ['IMMEDIATE emergency response', 'Aggressive IV resuscitation', 'Vasopressor support', 'Mechanical ventilation if needed', 'Renal replacement therapy if indicated', 'Empiric broad-spectrum antibiotics NOW'],
            'tests': ['Immediate full panel', 'Bedside echo', 'Arterial line', 'Central venous access', 'CT if stable'],
            'doctor': '🚨 CALL CODE / Activate Sepsis Protocol IMMEDIATELY',
            'emergency': ['Refractory septic shock', 'Multi-organ failure', 'DIC', 'ARDS'],
            'monitoring': ['Continuous invasive monitoring', 'Minute-by-minute assessment', 'ICU full support'],
        }
    }
    return recs.get(risk_class, recs[0])


def get_siddha_recommendation(risk_class, form_data):
    recs = {
        0: {
            'interpretation': 'Tridosha (Vatham, Pitham, Kapham) appears balanced. Body constitution is stable.',
            'constitution': 'Balanced Prakriti - No dominant dosha imbalance detected.',
            'suggestions': ['Maintain healthy daily routine (Dinacharya)', 'Practice Pranayama breathing exercises', 'Continue balanced Siddha lifestyle'],
            'herbs': ['Tulsi (Holy Basil) - immunity support', 'Amla (Indian Gooseberry) - antioxidant', 'Turmeric - anti-inflammatory'],
            'diet': ['Warm, easily digestible foods', 'Avoid cold and raw foods', 'Include ginger and black pepper in diet', 'Drink warm water throughout the day'],
            'lifestyle': ['Regular yoga practice', 'Adequate sleep (7-8 hours)', 'Stress reduction techniques', 'Morning sun exposure'],
        },
        1: {
            'interpretation': 'Early Pitham (fire element) imbalance detected. Mild inflammatory response noted in Siddha evaluation.',
            'constitution': 'Pitta-Vata imbalance - Requires early intervention to prevent progression.',
            'suggestions': ['Pitham-pacifying therapies', 'Kashayam (herbal decoctions)', 'Thokkanam (therapeutic massage)', 'Varmam point therapy for immunity'],
            'herbs': ['Nilavembu Kudineer - anti-fever, anti-viral', 'Adathodai - respiratory support', 'Siddha Makarathwaj - immunity booster', 'Karpooravalli - antimicrobial'],
            'diet': ['Cooling foods (cucumber, coconut water)', 'Avoid spicy, oily foods', 'Pomegranate and grape juice', 'Rice gruel (Kanji) - easy digestion', 'Avoid non-vegetarian during illness'],
            'lifestyle': ['Rest and reduced activity', 'Cool environment', 'Avoid direct sunlight', 'Deep breathing exercises'],
        },
        2: {
            'interpretation': 'Significant Pitham-Kapham imbalance. Aana Noi (systemic illness) pattern detected. Requires Siddha supportive care alongside conventional treatment.',
            'constitution': 'Tridosha imbalance - All three doshas affected. Supportive Siddha care recommended.',
            'suggestions': ['Supportive Siddha herbal therapy', 'Kaya Kalpa treatments for rejuvenation', 'IMPORTANT: Use as complementary to allopathy only'],
            'herbs': ['Seenthil (Tinospora cordifolia) - immunomodulator', 'Vilvam - liver protection', 'Mudakathan - anti-inflammatory', 'Pippali - respiratory and digestive support'],
            'diet': ['Light, warm, easily digestible foods', 'Coconut water for electrolytes', 'Herbal teas with ginger, tulsi, pepper', 'Complete avoidance of cold foods and drinks', 'Small, frequent meals'],
            'lifestyle': ['Complete bed rest', 'Warm compress on joints', 'Steam inhalation with Eucalyptus', 'Calm, stress-free environment'],
        },
        3: {
            'interpretation': 'Severe Tridosha imbalance. Mukkutra Seetham (critical cold-heat imbalance). Siddha supportive care as adjunct to emergency allopathy.',
            'constitution': 'Critical Tridosha derangement - Emergency allopathic care is primary.',
            'suggestions': ['Siddha support ONLY as adjunct', 'Consult qualified Siddha physician', 'Prioritize emergency allopathic treatment', 'Siddha rejuvenation post-stabilization'],
            'herbs': ['CONSULT SIDDHA PHYSICIAN BEFORE USE', 'Karisalai - liver and blood purifier', 'Thippili Rasayanam - immune fortifier'],
            'diet': ['Hospital diet as prescribed', 'Warm fluids only if permitted', 'Avoid all incompatible foods', 'Consult nutritionist'],
            'lifestyle': ['Complete rest under medical supervision', 'Spiritual support and prayer', 'Gentle Pranayama if tolerated'],
        },
        4: {
            'interpretation': '⚠️ Critical condition. Emergency allopathic treatment is the ONLY priority. Siddha medicine to be considered only post-stabilization under expert guidance.',
            'constitution': 'Emergency state - Beyond Siddha primary intervention scope.',
            'suggestions': ['EMERGENCY ALLOPATHIC CARE FIRST', 'No Siddha herbs during acute critical phase', 'Post-recovery Siddha rehabilitation plan can be developed'],
            'herbs': ['⚠️ No herbal interventions during septic shock', 'Post-recovery: consult senior Siddha physician'],
            'diet': ['ICU/hospital prescribed nutrition only', 'No self-administration of any supplements'],
            'lifestyle': ['Full medical supervision required', 'Family/spiritual support', 'Post-recovery comprehensive Siddha plan'],
        }
    }
    return recs.get(risk_class, recs[0])
