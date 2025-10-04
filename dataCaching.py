from datetime import datetime, timedelta
import json
import os

from app import fetch_celestrak_debris

CACHE_FILE = 'debris_cache.json'
CACHE_DURATION = timedelta(hours=6)  # Update every 6 hours

def get_cached_or_fetch():
    """Use cached data if available and fresh"""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
            cache_time = datetime.fromisoformat(cache['timestamp'])
            
            if datetime.now() - cache_time < CACHE_DURATION:
                return cache['data']
    
    # Fetch fresh data
    data = fetch_celestrak_debris()
    
    # Cache it
    with open(CACHE_FILE, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'data': data
        }, f)
    
    return data