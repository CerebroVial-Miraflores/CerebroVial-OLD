import os
import sys
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction.infrastructure.engine import TrafficModelEngine
from src.prediction.infrastructure.csv_loader import CSVLoader, FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Model Training...")
    
    # Paths
    data_dir = "data/traffic_logs"
    model_dir = "models"
    
    # Validation
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} not found.")
        return

    # 1. Load Data
    logger.info("Loading data logs...")
    loader = CSVLoader(data_dir=data_dir)
    df = loader.load_all_logs()
    
    if df.empty:
        logger.error("No data found in logs.")
        return
        
    logger.info(f"Loaded {len(df)} rows of data.")
    
    # 2. Feature Engineering
    logger.info("Preprocessing and creating targets...")
    engineer = FeatureEngineer()
    df = engineer.preprocess(df)
    
    # Create targets for different horizons
    horizons = [15, 30, 45]
    df = engineer.create_targets(df, time_windows=horizons)
    
    # Check data sufficiency
    logger.info(f"Data columns: {df.columns.tolist()}")
    
    # 3. Train Models
    engine = TrafficModelEngine(model_dir=model_dir)
    
    feature_cols = engineer.feature_columns + ['hour', 'day_of_week']
    
    # Train for each horizon
    # Current
    logger.info("Training 'Current' status model...")
    # For 'current', we might just need classification based on heuristic ground truth
    # But usually we predict future. Let's strictly follow the engine's capability.
    # The 'predict' method logic in Engine expects 'predicted_congestion_current', etc.
    # Actually, usually 'current' is just read-out, but if we want to "predict" it, 
    # it implies inferring congestion from raw metrics if we define it that way.
    # The CSVLoader creates 'target_current' which is the heuristic label.
    engine.train(df, feature_cols, 'target_current', 'total_vehicles', 'current')
    
    # Future Horizons
    for h in horizons:
        logger.info(f"Training {h}min horizon models...")
        target_class = f'target_congestion_{h}min'
        target_reg = f'target_vehicles_{h}min'
        
        engine.train(df, feature_cols, target_class, target_reg, f"{h}min")
        
    logger.info("Training Complete. Models saved to ./models/")

if __name__ == "__main__":
    main()
