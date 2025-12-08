import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_data(num_days=3):
    print("Generating synthetic traffic data...")
    
    # Time range
    start_time = datetime.now() - timedelta(days=num_days)
    end_time = datetime.now()
    
    # 1 minute intervals
    timestamps = pd.date_range(start=start_time, end=end_time, freq='1min')
    
    data = []
    
    # Simulate a daily pattern
    for ts in timestamps:
        hour = ts.hour
        is_weekend = ts.weekday() >= 5
        
        # Base traffic: peak hours 7-9 and 17-19
        base_traffic = 20
        
        if 7 <= hour <= 9:
            base_traffic = 80
        elif 17 <= hour <= 19:
            base_traffic = 90
        elif 10 <= hour <= 16:
            base_traffic = 50
        else:
            base_traffic = 10
            
        if is_weekend:
            base_traffic *= 0.6
            
        # Add noise
        total_vehicles = int(max(0, np.random.normal(base_traffic, 10)))
        
        # Derived metrics
        occupancy = min(1.0, total_vehicles / 100.0)
        flow = total_vehicles / 1.0 # per min
        speed = max(5, 60 - (occupancy * 50)) # slower if dense
        density = total_vehicles / 5.0 # arbitrary length
        
        # Congestion Label
        if speed < 15 or occupancy > 0.8:
            label = "Heavy"
        elif speed < 30 or occupancy > 0.5:
            label = "High"
        else:
            label = "Normal"
            
        row = {
            "timestamp": ts.timestamp(),
            "camera_id": "CAM_SYNTH",
            "total_vehicles": total_vehicles,
            "occupancy_rate": occupancy,
            "flow_rate_per_min": flow,
            "avg_speed": speed,
            "avg_density": density,
            "congestion_level": label,
            "hour": hour,
            "day_of_week": ts.weekday()
        }
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Save
    os.makedirs("data/traffic_logs", exist_ok=True)
    filename = "data/traffic_logs/traffic_log_synthetic_generated.csv"
    df.to_csv(filename, index=False)
    print(f"Generated {len(df)} rows to {filename}")

if __name__ == "__main__":
    generate_synthetic_data()
