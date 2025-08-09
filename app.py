# app.py
from flask import Flask, request, jsonify, render_template, send_from_directory
from pdf2image import convert_from_path
import pytesseract
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import numpy as np
from datetime import datetime
import time

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Medical patterns for extraction
MEDICAL_PATTERNS = {
    'rbc': [
        r'Red Blood Cell Count\s*[:=]?\s*(\d+\.\d+)',
        r'RBC\s*[:=]?\s*(\d+\.\d+)'
    ],
    'systolic': [
        r'Blood Pressure \(Systolic\)\s*[:=]?\s*(\d+)',
        r'Systolic\s*[:=]?\s*(\d+)'
    ],
    'diastolic': [
        r'Blood Pressure \(Diastolic\)\s*[:=]?\s*(\d+)',
        r'Diastolic\s*[:=]?\s*(\d+)'
    ],
    'glucose': [
        r'Glucose\s*[:=]?\s*(\d+)',
        r'Blood Sugar\s*[:=]?\s*(\d+)'
    ]
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_medical_data(text):
    results = {key: [] for key in MEDICAL_PATTERNS.keys()}
    for key, patterns in MEDICAL_PATTERNS.items():
        for pattern in patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                try:
                    results[key].append(float(match.group(1)))
                except ValueError:
                    continue
    return results

def create_visualizations(data, filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    visualizations = {}
    
    # Blood Pressure Chart
    if 'systolic' in data and 'diastolic' in data:
        plt.figure(figsize=(10, 6))
        plt.plot(data['systolic'], label='Systolic', marker='o', color='#3498db')
        plt.plot(data['diastolic'], label='Diastolic', marker='o', color='#e74c3c')
        plt.title('Blood Pressure Trend', fontsize=14)
        plt.xlabel('Measurement', fontsize=12)
        plt.ylabel('mmHg', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        bp_path = os.path.join(app.config['STATIC_FOLDER'], f'bp_{timestamp}.png')
        plt.savefig(bp_path, bbox_inches='tight', dpi=300)
        plt.close()
        visualizations['blood_pressure'] = f'/static/bp_{timestamp}.png'
    
    # RBC Chart
    if 'rbc' in data:
        plt.figure(figsize=(10, 6))
        plt.plot(data['rbc'], label='RBC Count', marker='o', color='#2ecc71')
        plt.title('Red Blood Cell Count Trend', fontsize=14)
        plt.xlabel('Measurement', fontsize=12)
        plt.ylabel('10^6/μL', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        rbc_path = os.path.join(app.config['STATIC_FOLDER'], f'rbc_{timestamp}.png')
        plt.savefig(rbc_path, bbox_inches='tight', dpi=300)
        plt.close()
        visualizations['rbc'] = f'/static/rbc_{timestamp}.png'
    
    # Glucose Chart
    if 'glucose' in data:
        plt.figure(figsize=(10, 6))
        plt.plot(data['glucose'], label='Glucose', marker='o', color='#9b59b6')
        plt.title('Blood Glucose Trend', fontsize=14)
        plt.xlabel('Measurement', fontsize=12)
        plt.ylabel('mg/dL', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        glucose_path = os.path.join(app.config['STATIC_FOLDER'], f'glucose_{timestamp}.png')
        plt.savefig(glucose_path, bbox_inches='tight', dpi=300)
        plt.close()
        visualizations['glucose'] = f'/static/glucose_{timestamp}.png'
    
    return visualizations

def generate_report(data, filename):
    report = {
        'filename': filename,
        'date_analyzed': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'statistics': {},
        'measurements': []
    }
    
    # Calculate statistics
    for key, values in data.items():
        if values:
            report['statistics'][key] = {
                'average': round(np.mean(values), 2),
                'min': round(min(values), 2),
                'max': round(max(values), 2),
                'count': len(values)
            }
    
    # Prepare measurement data
    max_measurements = max(len(values) for values in data.values() if values)
    for i in range(max_measurements):
        measurement = {}
        for key in data.keys():
            if i < len(data[key]):
                measurement[key] = data[key][i]
            else:
                measurement[key] = None
        report['measurements'].append(measurement)
    
    return report

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Only PDF files are accepted'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Convert PDF to images
        start_time = time.time()
        images = convert_from_path(filepath, dpi=300)
        
        # Extract text from images
        full_text = ""
        for i, image in enumerate(images):
            text = pytesseract.image_to_string(image)
            full_text += f"--- Page {i+1} ---\n{text}\n"
        
        # Extract medical data
        medical_data = extract_medical_data(full_text)
        
        if not any(medical_data.values()):
            return jsonify({
                'error': 'No medical data found in the document',
                'debug_text': full_text[:1000] + ("..." if len(full_text) > 1000 else "")
            }), 400
        
        # Create visualizations
        visualizations = create_visualizations(medical_data, filename)
        
        # Generate report
        report = generate_report(medical_data, filename)
        
        return jsonify({
            'status': 'success',
            'report': report,
            'visualizations': visualizations,
            'processing_time': round(time.time() - start_time, 2)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)