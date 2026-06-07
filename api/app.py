"""
FastAPI application for the Stock Transformer prediction service.
Uses the modern lifespan context manager to load the model on startup and
clean up on shutdown (if needed).
Endpoints:
    GET  /health              – service health check
    POST /predict             – single‑step prediction for a given ticker
    POST /predict/multistep   – multi‑step autoregressive forecast
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import sys

# Add project root to path so 'src' can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference import InferenceEngine
from src.logger import logger


# Lifespan – loads the engine once at startup
engine: Optional[InferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("Loading inference engine...")
    engine = InferenceEngine()          # uses config.yaml by default
    logger.info("Inference engine loaded successfully.")
    yield                               # application runs here
    # Optional cleanup could go here
    logger.info("Shutting down inference engine.")


app = FastAPI(title="Stock Transformer API", version="1.0.0", lifespan=lifespan)




# Request / Response models
class SingleStepResponse(BaseModel):
    ticker: str
    next_price: float
    price_uncertainty: float
    direction: str
    direction_probs: dict
    volatility: float


class MultiStepResponse(BaseModel):
    ticker: str
    horizon: int
    prices: List[float]
    uncertainties: List[float]
    directions: List[str]



# Endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=SingleStepResponse)
def predict_single(ticker: str = Query(..., description="Stock ticker, e.g. AAPL.US")):
    """Return the next‑day price, direction, and volatility."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")
    try:
        # Load the ticker's data and take the most recent window
        df = engine._load_ticker_data(ticker)
        split_idx = int(len(df) * 0.9)
        train_df = df.iloc[:split_idx]
        recent_window = train_df.iloc[-engine.seq_len:]
        result = engine.single_step_predict(ticker, recent_window)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"ticker": ticker, **result}


@app.post("/predict/multistep", response_model=MultiStepResponse)
def predict_multistep(
    ticker: str = Query(..., description="Stock ticker, e.g. AAPL.US"),
    horizon: int = Query(21, description="Number of days to forecast")
):
    """Generate a multi‑step autoregressive forecast."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialised yet.")
    try:
        df = engine._load_ticker_data(ticker)
        split_idx = int(len(df) * 0.9)
        train_df = df.iloc[:split_idx]
        recent_window = train_df.iloc[-engine.seq_len:]
        result = engine.autoregressive_predict(ticker, recent_window, horizon=horizon)
    except Exception as e:
        logger.error(f"Multi‑step prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    return {"ticker": ticker, "horizon": horizon, **result}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)