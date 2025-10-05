from flask import Flask, render_template, request, jsonify
import os
from ultralytics import YOLO
import cv2
import requests
from datetime import datetime, timedelta
import json

app = Flask(__name__)

# Paths
MODEL_PATH = 'best.pt'
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
CACHE_FILE = 'debris_cache.json'
CACHE_DURATION = timedelta(hours=6)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model once at the beginning
model = YOLO(MODEL_PATH)

# ============= HELPER FUNCTIONS =============

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
        print(f"Fetching data from CelesTrak: {base_url} with params {params}")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        if format == 'json':
            data = response.json()
            print(f"Successfully fetched {len(data)} objects from CelesTrak")
            return data
        elif format == 'csv':
            return response.text
        else:
            return response.text
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from CelesTrak: {e}")
        return None

def get_cached_or_fetch(group='analyst'):
    """Use cached data if available and fresh, otherwise fetch new data"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache = json.load(f)
                cache_time = datetime.fromisoformat(cache['timestamp'])
                
                if datetime.now() - cache_time < CACHE_DURATION:
                    print("Using cached data")
                    return cache['data']
        except Exception as e:
            print(f"Error reading cache: {e}")
    
    # Fetch fresh data
    print("Fetching fresh data from CelesTrak...")
    data = fetch_celestrak_data(group=group, format='json')
    
    if data:
        # Cache it
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'data': data
                }, f)
            print("Data cached successfully")
        except Exception as e:
            print(f"Error caching data: {e}")
    
    return data

def calculate_altitude(obj):
    """Calculate approximate altitude from TLE data"""
    try:
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
        import math
        a = (mu * (period_seconds / (2 * math.pi))**2)**(1/3)
        
        # Altitude = semi-major axis - Earth radius
        altitude = a - earth_radius
        
        return round(altitude, 2)
    except Exception as e:
        print(f"Error calculating altitude: {e}")
        return None

def filter_debris_only(data):
    """Filter to only debris objects"""
    debris = []
    for obj in data:
        obj_type = obj.get('OBJECT_TYPE', '').upper()
        obj_name = obj.get('OBJECT_NAME', '').upper()
        
        # Include DEBRIS, ROCKET BODY, and objects with DEB in name as space junk
        if ('DEBRIS' in obj_type or 
            'ROCKET BODY' in obj_type or 
            'DEB' in obj_type or 
            'DEB' in obj_name or
            'R/B' in obj_name):
            debris.append(obj)
    
    print(f"Filtered {len(debris)} debris objects from {len(data)} total objects")
    return debris

def parse_satellite_data(data):
    """Parse and simplify satellite/debris data for young learners"""
    simplified = []
    
    for obj in data:
        try:
            # CelesTrak JSON format uses these field names
            satellite_info = {
                'name': obj.get('OBJECT_NAME', 'Unknown'),
                'satellite_id': obj.get('NORAD_CAT_ID'),
                'country': obj.get('OBJECT_ID', 'UN')[:2],  # First 2 chars indicate country
                'launch_date': obj.get('EPOCH'),
                'altitude_km': calculate_altitude(obj),
                'inclination': obj.get('INCLINATION'),
                'period_minutes': obj.get('PERIOD'),
                'object_type': obj.get('OBJECT_TYPE', 'UNKNOWN')
            }
            simplified.append(satellite_info)
        except Exception as e:
            print(f"Error parsing object: {e}")
            continue
    
    return simplified

# ============= ROUTES =============

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/api/debris')
def get_debris_data():
    """API endpoint to fetch debris data"""
    try:
        # Fetch data with caching
        all_data = get_cached_or_fetch(group='analyst')
        
        if not all_data:
            print("No data returned from CelesTrak")
            return jsonify({'error': 'Could not fetch data from CelesTrak'}), 500
        
        print(f"Processing {len(all_data)} total objects")
        
        # Filter to only debris
        debris = filter_debris_only(all_data)
        
        if not debris:
            # If no debris found with strict filtering, return all data
            print("No debris found with filtering, using all data")
            debris = all_data
        
        # Parse the data
        simplified = parse_satellite_data(debris)
        
        print(f"Returning {len(simplified)} debris objects")
        
        return jsonify({
            'count': len(simplified),
            'debris': simplified[:100],  # Limit to 100 for performance
            'total_tracked': len(all_data)
        })
        
    except Exception as e:
        print(f"Error in get_debris_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/stats')
def get_debris_stats():
    """Get statistics about space debris"""
    try:
        # Fetch data with caching
        all_data = get_cached_or_fetch(group='analyst')
        
        if not all_data:
            return jsonify({'error': 'Could not fetch data from CelesTrak'}), 500
        
        # Filter debris
        debris = filter_debris_only(all_data)
        
        if not debris:
            # If no debris found, use all data for stats
            debris = all_data
        
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
            try:
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
            except Exception as e:
                print(f"Error processing object for stats: {e}")
                continue
        
        print(f"Stats calculated: {total} debris, {len(countries)} countries")
        
        return jsonify({
            'total_debris': total,
            'by_country': countries,
            'by_altitude': altitude_ranges,
            'by_type': object_types
        })
        
    except Exception as e:
        print(f"Error in get_debris_stats: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predict', methods=["POST", "GET"])
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
    result_img = f"results/{result_filename}"   # relative to static
    original_file_path = f"uploads/{file.filename}"
    return render_template("index.html", result_img=result_img, upload_img=original_file_path)

# Test endpoint to verify API is working
@app.route('/api/test')
def test_api():
    """Test endpoint to check if CelesTrak API is accessible"""
    try:
        data = fetch_celestrak_data(group='active', format='json')
        if data:
            return jsonify({
                'status': 'success',
                'message': 'CelesTrak API is accessible',
                'sample_count': len(data),
                'sample_object': data[0] if data else None
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Could not fetch data from CelesTrak'
            }), 500
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)