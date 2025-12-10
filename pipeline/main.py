"""
Main orchestration script for WISDM51 data processing pipeline.

Pipeline flow:
1. Load raw sensor data from all devices/sensors
2. Clean data (handle NaN, inf, stuck sensors)
3. Create sliding windows with class consistency validation
4. Extract handcrafted time-domain features
5. Save final feature matrix as CSV

Usage:
    python main.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from time import time

# Import pipeline modules
from config import (
    OUTPUT_DIR, DATA_DIR, CLEANED_CSV, WINDOWED_CSV, FINAL_CSV,
    TIME_DOMAIN_FEATURES, WINDOW_SAMPLES
)
from utils import (
    setup_logging, load_all_raw_data, print_data_summary, 
    validate_dataframe
)
from cleaning import clean_data, get_cleaning_report
from windowing import create_windows, get_windowing_report
from features import extract_all_features, get_feature_extraction_report

logger = setup_logging()


def run_pipeline():
    """
    Execute the complete data processing pipeline.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # ======================== SETUP ========================
        logger.info("=" * 80)
        logger.info("WISDM51 DATA PROCESSING PIPELINE")
        logger.info("=" * 80)
        
        # Ensure output directories exist
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # ======================== STEP 1: LOAD RAW DATA ========================
        logger.info("\n[STEP 1] Loading raw sensor data...")
        start_time = time()
        
        try:
            df_raw = load_all_raw_data()
        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            return False
        
        print_data_summary(df_raw)
        logger.info(f"Time elapsed: {time() - start_time:.2f}s\n")
        
        # ======================== STEP 2: CLEAN DATA ========================
        logger.info("[STEP 2] Cleaning raw sensor data...")
        start_time = time()
        
        try:
            df_cleaned = clean_data(
                df_raw,
                handle_missing=True,
                handle_stuck=True,
                handle_outliers=False
            )
        except Exception as e:
            logger.error(f"Cleaning failed: {e}")
            return False
        
        # Save cleaned data
        df_cleaned.to_csv(CLEANED_CSV, index=False)
        logger.info(f"Saved cleaned data to: {CLEANED_CSV}")
        get_cleaning_report(df_raw, df_cleaned)
        logger.info(f"Time elapsed: {time() - start_time:.2f}s\n")
        
        # ======================== STEP 3: CREATE WINDOWS ========================
        logger.info("[STEP 3] Creating sliding windows...")
        start_time = time()
        
        try:
            df_windowed = create_windows(
                df_cleaned,
                window_samples=WINDOW_SAMPLES,
                overlap=0.5,
                class_consistency_threshold=0.80
            )
        except Exception as e:
            logger.error(f"Windowing failed: {e}")
            return False
        
        if df_windowed.empty:
            logger.error("No windows created! Check data and parameters.")
            return False
        
        # Save windowed data
        df_windowed.to_csv(WINDOWED_CSV, index=False)
        logger.info(f"Saved windowed data to: {WINDOWED_CSV}")
        get_windowing_report(df_cleaned, df_windowed)
        logger.info(f"Time elapsed: {time() - start_time:.2f}s\n")
        
        # ======================== STEP 4: EXTRACT FEATURES ========================
        logger.info("[STEP 4] Extracting handcrafted features...")
        start_time = time()
        
        try:
            df_features = extract_all_features(
                df_windowed,
                features_to_compute=TIME_DOMAIN_FEATURES,
                window_samples=WINDOW_SAMPLES
            )
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            logger.exception(e)
            return False
        
        if df_features.empty:
            logger.error("No features extracted!")
            return False
        
        # Save final feature matrix
        df_features.to_csv(FINAL_CSV, index=False)
        logger.info(f"Saved final feature matrix to: {FINAL_CSV}")
        get_feature_extraction_report(df_features, TIME_DOMAIN_FEATURES)
        logger.info(f"Time elapsed: {time() - start_time:.2f}s\n")
        
        # ======================== SUMMARY ========================
        logger.info("=" * 80)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Raw data samples: {len(df_raw)}")
        logger.info(f"Cleaned data samples: {len(df_cleaned)}")
        logger.info(f"Windows created: {len(df_windowed)}")
        logger.info(f"Feature matrix shape: {df_features.shape}")
        logger.info(f"\nOutputs saved:")
        logger.info(f"  - Cleaned data: {CLEANED_CSV}")
        logger.info(f"  - Windowed data: {WINDOWED_CSV}")
        logger.info(f"  - Feature matrix: {FINAL_CSV}")
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!\n")
        
        return True
        
    except Exception as e:
        logger.error(f"Unexpected error in pipeline: {e}")
        logger.exception(e)
        return False


if __name__ == '__main__':
    success = run_pipeline()
    sys.exit(0 if success else 1)
