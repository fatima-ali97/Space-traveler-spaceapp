from flask import Flask, render_template, request, jsonify
import os
from ultralytics import YOLO
import cv2
import requests
from datetime import datetime


app = Flask(__name__)

# Paths
MODEL_PATH = 'best.pt'
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model once at the beginning
model = YOLO(MODEL_PATH)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/debris')
def get_debris_data():
    """API endpoint to fetch debris data"""
    # Fetch analyst satellites which include debris tracking
    all_data = fetch_celestrak_data(group='analyst', format='json')
    
    if all_data:
        # Filter to only debris
        debris = filter_debris_only(all_data)
        simplified = parse_satellite_data(debris)
        
        return jsonify({
            'count': len(simplified),
            'debris': simplified[:100],  # Limit to 100 for performance
            'total_tracked': len(all_data)
        })
    return jsonify({'error': 'Could not fetch data'}), 500

@app.route('/api/stats')
def get_debris_stats():
    """Get statistics about space debris"""
    all_data = fetch_celestrak_data(group='analyst', format='json')
    
    if not all_data:
        return jsonify({'error': 'Could not fetch data'}), 500
    
    # Filter debris
    debris = filter_debris_only(all_data)
    
    # Calculate statistics
    total = len(debris)
    countries = {}
    altitude_ranges = {
        'Low (0-2000 km)': 0,
        'Medium (2000-35000 km)': 0,
        'High (35000+ km)': 0
    }
    
    object_types = {}
    
    for obj in debris:
        # Count by country
        country_code = obj.get('OBJECT_ID', 'UN')[:2]
        countries[country_code] = countries.get(country_code, 0) + 1
        
        # Count by type
        obj_type = obj.get('OBJECT_TYPE', 'UNKNOWN')
        object_types[obj_type] = object_types.get(obj_type, 0) + 1
        
        # Count by altitude
        altitude = calculate_altitude(obj)
        if altitude:
            if altitude < 2000:
                altitude_ranges['Low (0-2000 km)'] += 1
            elif altitude < 35000:
                altitude_ranges['Medium (2000-35000 km)'] += 1
            else:
                altitude_ranges['High (35000+ km)'] += 1
    
    return jsonify({
        'total_debris': total,
        'by_country': countries,
        'by_altitude': altitude_ranges,
        'by_type': object_types
    })



# NOOO
def fetch_celestrak_by_name(name, format='json'):
    """Fetch data by searching satellite name"""
    base_url = "https://celestrak.org/NORAD/elements/gp.php"
    params = {
        'NAME': name,
        'FORMAT': format
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json() if format == 'json' else response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def parse_satellite_data(data):
    """Parse and simplify satellite/debris data for young learners"""
    simplified = []
    
    for obj in data:
        # CelesTrak JSON format uses these field names
        satellite_info = {
            'name': obj.get('OBJECT_NAME', 'Unknown'),
            'satellite_id': obj.get('NORAD_CAT_ID'),
            'country': obj.get('OBJECT_ID', '')[:2],  # First 2 chars indicate country
            'launch_date': obj.get('EPOCH'),
            'altitude_km': calculate_altitude(obj),
            'inclination': obj.get('INCLINATION'),
            'period_minutes': obj.get('PERIOD'),
            'object_type': obj.get('OBJECT_TYPE', 'UNKNOWN')  # PAYLOAD, ROCKET BODY, DEBRIS, etc.
        }
        simplified.append(satellite_info)
    
    return simplified

def filter_debris_only(data):
    """Filter to only debris objects"""
    debris = []
    for obj in data:
        obj_type = obj.get('OBJECT_TYPE', '').upper()
        # Include DEBRIS and old ROCKET BODY as space junk
        if 'DEBRIS' in obj_type or 'ROCKET BODY' in obj_type or 'DEB' in obj_type:
            debris.append(obj)
    return debris

def calculate_altitude(obj):
    """Calculate approximate altitude from TLE data"""
    # Mean motion is revolutions per day
    mean_motion = float(obj.get('MEAN_MOTION', 0))
    
    if mean_motion == 0:
        return None
    
    # Calculate semi-major axis using Kepler's third law
    earth_radius = 6371  # km
    mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
    
    # Period in minutes
    period = 1440 / mean_motion  # 1440 minutes in a day
    period_seconds = period * 60
    
    # Semi-major axis
    a = (mu * (period_seconds / (2 * 3.14159))**2)**(1/3)
    
    # Altitude = semi-major axis - Earth radius
    altitude = a - earth_radius
    
    return round(altitude, 2)

# yes
def fetch_celestrak_data(group='analyst', format='json'):
    """
    Fetch orbital data from CelesTrak
    
    Args:
        group: GROUP parameter (e.g., 'analyst', 'active', 'starlink')
        format: FORMAT parameter (json, xml, csv, tle)
    """
    base_url = "https://celestrak.org/NORAD/elements/gp.php"
    params = {
        'GROUP': group,
        'FORMAT': format
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        if format == 'json':
            return response.json()
        elif format == 'csv':
            return response.text
        else:
            return response.text
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def fetch_celestrak_debris():
    """Fetch space debris data from CelesTrak"""
    url = "https://celestrak.org/NORAD/elements/gp-first.php?NAME=COSMOS%202251%20DEB"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/predict', methods=["POST","GET"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Run YOLO model
    results = model(file_path)[0]
    conf_threshold = 0.6

    # Read uploaded image
    img = cv2.imread(file_path)

    # Loop over detections
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = f"{model.names[cls_id]} {conf:.2f}"
            # Draw box and label
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save result image
    result_filename = f"result_{file.filename}"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, img)

    print(f"Saved result to {result_path}")
    result_filename = f"result_{file.filename}"
    result_img = f"results/{result_filename}"   # relative to static
    original_file_path = f"uploads/{file.filename}"
    return render_template("index.html", result_img=result_img, upload_img=original_file_path)

def calculate_altitude(obj):
    """Calculate approximate altitude from TLE data"""
    # Mean motion is revolutions per day
    mean_motion = float(obj.get('MEAN_MOTION', 0))
    
    if mean_motion == 0:
        return None
    
    # Calculate semi-major axis using Kepler's third law
    earth_radius = 6371  # km
    mu = 398600.4418  # Earth's gravitational parameter (km^3/s^2)
    
    # Period in minutes
    period = 1440 / mean_motion  # 1440 minutes in a day
    period_seconds = period * 60
    
    # Semi-major axis
    a = (mu * (period_seconds / (2 * 3.14159))**2)**(1/3)
    
    # Altitude = semi-major axis - Earth radius
    altitude = a - earth_radius
    
    return round(altitude, 2)

if __name__ == '__main__':
    app.run(debug=True)


