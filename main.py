"""
Eco-Pulse Backend - Main Application
FastAPI server for the energy audit dashboard.

Supports disk caching so the 10-minute pipeline only runs once.
Subsequent starts load from cache in ~5 seconds.
"""

import os
import sys
import time
import pickle
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Add backend to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from models.data_processor import DataProcessor
from models.change_point import ChangePointModel, fit_all_meters
from models.aggregator import BuildingAggregator
from api.routes import router, set_global_state


# Global state
data_processor = None
model_results = None
aggregator = None
merged_df = None

# ====================================================================
#  Cache helpers
# ====================================================================
CACHE_DIR_NAME = "cache"

def _cache_dir(data_dir: str) -> Path:
    """Return the cache directory under backend/data/processed/cache/."""
    return Path(__file__).parent / "data" / "processed" / CACHE_DIR_NAME


def _cache_paths(data_dir: str) -> dict:
    """Return a dict of cache file paths."""
    d = _cache_dir(data_dir)
    return {
        "merged_df":      d / "merged_df.pkl",
        "model_results":  d / "model_results.pkl",
        "building_df":    d / "building_df.pkl",
        "aggregator":     d / "aggregator.pkl",
    }


def _newest_source_mtime(data_dir: str) -> float:
    """
    Return the most-recent modification time among the raw CSV source files.
    If *any* source file is newer than the cache, we must recompute.
    """
    data_path = Path(data_dir)
    mtimes = []
    for pattern in ["meter-readings-*.csv", "building_metadata.csv", "weather_data_hourly_2025.csv"]:
        for f in data_path.glob(pattern):
            mtimes.append(f.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def _cache_is_valid(data_dir: str) -> bool:
    """Check whether all cache files exist and are newer than source data."""
    paths = _cache_paths(data_dir)
    if not all(p.exists() for p in paths.values()):
        return False
    source_mtime = _newest_source_mtime(data_dir)
    oldest_cache = min(p.stat().st_mtime for p in paths.values())
    return oldest_cache > source_mtime


def _save_cache(data_dir: str, merged_df_val, model_results_val, building_df_val, aggregator_val):
    """Persist pipeline outputs to disk."""
    d = _cache_dir(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = _cache_paths(data_dir)
    
    t0 = time.time()
    with open(paths["merged_df"], "wb") as f:
        pickle.dump(merged_df_val, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["model_results"], "wb") as f:
        pickle.dump(model_results_val, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["building_df"], "wb") as f:
        pickle.dump(building_df_val, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(paths["aggregator"], "wb") as f:
        pickle.dump(aggregator_val, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    total_size = sum(p.stat().st_size for p in paths.values()) / (1024 * 1024)
    print(f"\nCache saved to {d}  ({total_size:.1f} MB, {time.time() - t0:.1f}s)")


def _load_cache(data_dir: str):
    """Load pipeline outputs from disk. Returns (merged_df, model_results, building_df, aggregator)."""
    paths = _cache_paths(data_dir)
    t0 = time.time()
    
    with open(paths["merged_df"], "rb") as f:
        merged_df_val = pickle.load(f)
    with open(paths["model_results"], "rb") as f:
        model_results_val = pickle.load(f)
    with open(paths["building_df"], "rb") as f:
        building_df_val = pickle.load(f)
    with open(paths["aggregator"], "rb") as f:
        aggregator_val = pickle.load(f)
    
    elapsed = time.time() - t0
    print(f"  Cache loaded in {elapsed:.1f}s")
    return merged_df_val, model_results_val, building_df_val, aggregator_val


def clear_cache(data_dir: str):
    """Delete all cache files."""
    paths = _cache_paths(data_dir)
    for p in paths.values():
        if p.exists():
            p.unlink()
    print("Cache cleared.")


# ====================================================================
#  Initialization
# ====================================================================

def initialize_data(data_dir: str, use_cache: bool = True):
    """
    Initialize all data processing, model fitting, and aggregation.
    
    If use_cache is True and a valid cache exists, loads from disk
    in ~5 seconds instead of rerunning the full ~10 min pipeline.

    Args:
        data_dir: Path to directory containing raw CSV files
        use_cache: Whether to use disk cache (default True)
    """
    global data_processor, model_results, aggregator, merged_df
    
    print("=" * 60)
    print("ECO-PULSE BACKEND INITIALIZATION")
    print("=" * 60)
    
    # ------------------------------------------------------------------
    # Try loading from cache
    # ------------------------------------------------------------------
    if use_cache and _cache_is_valid(data_dir):
        print("\n  Valid cache found -- loading from disk...")
        merged_df, model_results, building_df, aggregator = _load_cache(data_dir)
        
        # Reconstruct a lightweight DataProcessor for the API
        # (get_data_summary needs merged_df)
        data_processor = DataProcessor(data_dir)
        data_processor.merged_df = merged_df
        data_processor.building_df = building_df
        
        summary = aggregator.get_summary_statistics()
        print(f"\n  Buildings: {summary['total_buildings']}")
        eff_mean = summary['overall_efficiency']['mean']
        print(f"  Efficiency mean: {eff_mean:.1f}" if eff_mean else "  Efficiency mean: N/A")
        print(f"  Confidence: {summary['confidence_distribution']}")
        
        set_global_state(data_processor, model_results, aggregator, merged_df)
        return data_processor, model_results, aggregator
    
    if use_cache:
        print("\n  No valid cache found -- running full pipeline...")
    else:
        print("\n  Cache disabled -- running full pipeline...")
    
    # ------------------------------------------------------------------
    # Phase 1: Data Processing
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 1: DATA PROCESSING")
    print("=" * 60)
    
    data_processor = DataProcessor(data_dir)
    data_processor.load_energy_data()
    data_processor.load_building_metadata()
    data_processor.load_and_process_weather()
    merged_df = data_processor.join_all_data()
    
    # ------------------------------------------------------------------
    # Phase 2: Change-Point Model Fitting
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2: CHANGE-POINT MODEL FITTING")
    print("=" * 60)
    
    model_results = fit_all_meters(merged_df)
    
    # ------------------------------------------------------------------
    # Phase 3: Building Aggregation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 3: BUILDING AGGREGATION")
    print("=" * 60)
    
    aggregator = BuildingAggregator(data_processor.building_df)
    aggregator.aggregate_all_buildings(model_results, merged_df)
    
    # ------------------------------------------------------------------
    # Save cache for next time
    # ------------------------------------------------------------------
    _save_cache(data_dir, merged_df, model_results, data_processor.building_df, aggregator)
    
    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("INITIALIZATION COMPLETE")
    print("=" * 60)
    
    summary = aggregator.get_summary_statistics()
    print(f"\nTotal buildings analyzed: {summary['total_buildings']}")
    eff_mean = summary['overall_efficiency']['mean']
    print(f"Overall efficiency mean: {eff_mean:.1f}" if eff_mean else "Overall efficiency mean: N/A")
    print(f"Confidence distribution: {summary['confidence_distribution']}")
    if summary.get('quality_flag_counts'):
        print(f"Quality flag counts: {summary['quality_flag_counts']}")
    
    # Set global state for API
    set_global_state(data_processor, model_results, aggregator, merged_df)
    
    return data_processor, model_results, aggregator


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Initializes data on startup.
    """
    data_dir = os.environ.get(
        'ECOPULSE_DATA_DIR',
        str(Path(__file__).parent.parent)
    )
    use_cache = os.environ.get('ECOPULSE_NO_CACHE', '').lower() not in ('1', 'true', 'yes')
    
    print(f"\nData directory: {data_dir}")
    print(f"Cache enabled: {use_cache}")
    
    initialize_data(data_dir, use_cache=use_cache)
    
    yield
    
    print("\nShutting down Eco-Pulse backend...")


# Create FastAPI app
app = FastAPI(
    title="Eco-Pulse API",
    description="""
    ## Physics-Informed Energy Audit Dashboard
    
    Eco-Pulse analyzes building energy efficiency using ASHRAE Change-Point Regression.
    
    ### Features
    - **Building Rankings**: View all buildings ranked by efficiency
    - **V-Curve Visualization**: See energy vs temperature relationships
    - **Multi-Meter Analysis**: Aggregate scores across heating and cooling systems
    - **Change-Point Detection**: Identify building balance temperatures
    
    ### Methodology
    - Uses 3-parameter (3P) and 4-parameter (4P) change-point models
    - Calculates Heating Degree Days (HDD) and Cooling Degree Days (CDD)
    - Produces efficiency scores from 0-100 (higher = more efficient)
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    return {
        "name": "Eco-Pulse API",
        "version": "1.0.0",
        "description": "Physics-Informed Energy Audit Dashboard",
        "docs": "/docs",
        "health": "/api/health"
    }


# CLI entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Eco-Pulse Backend Server")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent),
        help="Directory containing raw CSV files"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force full pipeline recomputation (ignore cache)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Delete existing cache and exit"
    )
    
    args = parser.parse_args()
    
    # Set environment variable for data directory
    os.environ['ECOPULSE_DATA_DIR'] = args.data_dir
    
    if args.clear_cache:
        clear_cache(args.data_dir)
        sys.exit(0)
    
    if args.no_cache:
        os.environ['ECOPULSE_NO_CACHE'] = '1'
    
    print(f"\nStarting Eco-Pulse backend server...")
    print(f"Data directory: {args.data_dir}")
    print(f"Cache: {'disabled' if args.no_cache else 'enabled'}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
