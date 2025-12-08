import pandas as pd
import glob
import os
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class CSVLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_all_logs(self, pattern: str = "traffic_log_*.csv") -> pd.DataFrame:
        """Loads all CSV logs matching the pattern from the data directory."""
        search_path = os.path.join(self.data_dir, pattern)
        files = glob.glob(search_path)
        
        if not files:
            logger.warning(f"No files found for pattern {search_path}")
            return pd.DataFrame()
            
        dfs = []
        for file in files:
            try:
                # Read CSV
                df = pd.read_csv(file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                
        if not dfs:
            return pd.DataFrame()
            
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Sort by timestamp to ensure order
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values('timestamp')
            
        return combined_df

    def get_latest_log_entry(self, camera_id: str) -> Optional[dict]:
        """
        Get the specific last entry for a camera from all logs.
        """
        try:
            df = self.load_all_logs()
            
            if df.empty:
                return None
                
            if 'camera_id' not in df.columns:
                return None
                
            camera_data = df[df['camera_id'] == camera_id]
            
            if camera_data.empty:
                return None
            
            # Since load_all_logs returns sorted (if timestamp exists), tail(1) is the latest
            return camera_data.iloc[-1].to_dict()
            
        except Exception as e:
            logger.error(f"Error getting latest entry: {e}")
            return None

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = [
            'total_vehicles', 
            'occupancy_rate', 
            'flow_rate_per_min', 
            'avg_speed', 
            'avg_density'
        ]
        
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and prepares the raw DataFrame."""
        df = df.copy()
        
        # Ensure numeric
        for col in self.feature_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=self.feature_columns)
        
        # Add time features if timestamp exists
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            
        return df

    def create_targets(self, df: pd.DataFrame, time_windows: List[int] = [15, 30, 45]) -> pd.DataFrame:
        """
        Creates target variables for future prediction.
        """
        if 'camera_id' in df.columns and 'timestamp' in df.columns:
            df = df.sort_values(['camera_id', 'timestamp'])
        
        def calculate_congestion(row):
            if 'congestion_level' in row and pd.notna(row['congestion_level']):
                return row['congestion_level']

            # Heuristic Correction
            if row['avg_density'] <= 0.5:
                # Very low density is Normal
                return "Normal"

            if row['avg_speed'] < 10 or row['avg_density'] > 8:
                return "Heavy"
            elif row['avg_speed'] < 25 or row['avg_density'] > 4:
                return "High"
            else:
                return "Normal"

        # Apply heuristic if target not present
        if 'congestion_label' not in df.columns:
            df['congestion_label'] = df.apply(calculate_congestion, axis=1)
        
        label_map = {"Normal": 0, "High": 1, "Heavy": 2}
        df['target_current'] = df['congestion_label'].map(label_map)
        
        for mins in time_windows:
            # Approx 2-3 seconds per sample. 
            # 15 mins * 60 = 900s. 900 / 3 = 300 steps.
            shift_steps = int((mins * 60) / 3) 
            
            if 'camera_id' in df.columns:
                df[f'target_congestion_{mins}min'] = df.groupby('camera_id')['target_current'].shift(-shift_steps)
                df[f'target_vehicles_{mins}min'] = df.groupby('camera_id')['total_vehicles'].shift(-shift_steps)
            else:
                 df[f'target_congestion_{mins}min'] = df['target_current'].shift(-shift_steps)
                 df[f'target_vehicles_{mins}min'] = df['total_vehicles'].shift(-shift_steps)

        return df
