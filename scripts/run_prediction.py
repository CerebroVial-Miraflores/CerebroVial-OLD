import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.prediction.presentation.api.routes import router as prediction_router, init_predictor
from src.prediction.application.predictor import CongestionPredictor

def create_app() -> FastAPI:
    app = FastAPI(title="CerebroVial Prediction API")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize Predictor
    print("Initializing Predictor Service...")
    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/traffic_logs", exist_ok=True)
    
    predictor = CongestionPredictor(model_dir="models", data_dir="data/traffic_logs")
    
    # Inject dependency into router
    init_predictor(predictor)
    
    app.include_router(prediction_router)
    
    return app

def main():
    print("Starting Prediction Server...")
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8001)

if __name__ == "__main__":
    main()
