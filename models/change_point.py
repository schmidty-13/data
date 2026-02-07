"""
Change-Point Model Module for Eco-Pulse
Implements ASHRAE 3-parameter and 4-parameter change-point regression models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')


# Utility type to degree-days mapping
UTILITY_MODELS = {
    'HEAT': {'degree_days': 'HDD', 'interpretation': 'Heating efficiency'},
    'STEAM': {'degree_days': 'HDD', 'interpretation': 'Steam heating efficiency'},
    'GAS': {'degree_days': 'HDD', 'interpretation': 'Gas heating efficiency'},
    'ELECTRICITY': {'degree_days': 'CDD', 'interpretation': 'Cooling/electrical efficiency'},
    'ELECTRICAL_POWER': {'degree_days': 'CDD', 'interpretation': 'Electrical power efficiency'},
    'COOLING': {'degree_days': 'CDD', 'interpretation': 'Cooling efficiency'},
    'COOLING_POWER': {'degree_days': 'CDD', 'interpretation': 'Cooling power efficiency'},
}


@dataclass
class ModelResult:
    """
    Results from a change-point model fit.
    
    Attributes:
        model_type: '3P' or '4P'
        t_balance: Change-point temperature (°F)
        base_load: Base energy load (intercept, β₀)
        heating_slope: Heating slope (β₁), energy per HDD
        cooling_slope: Cooling slope (β₂), energy per CDD (4P only)
        r2: R-squared (coefficient of determination)
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        n_observations: Number of data points
        confidence: 'high', 'medium', or 'low' based on R²
        predictions: Predicted energy values
        residuals: Actual - Predicted values
        validation_flags: List of validation warnings (Task 5.2)
    """
    model_type: str
    t_balance: float
    base_load: float
    heating_slope: Optional[float]
    cooling_slope: Optional[float]
    r2: float
    rmse: float
    mae: float
    n_observations: int
    confidence: str
    predictions: np.ndarray
    residuals: np.ndarray
    validation_flags: list = None
    
    def __post_init__(self):
        if self.validation_flags is None:
            self.validation_flags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding arrays for JSON serialization)."""
        return {
            'model_type': self.model_type,
            't_balance': round(self.t_balance, 2),
            'base_load': round(self.base_load, 4),
            'heating_slope': round(self.heating_slope, 4) if self.heating_slope else None,
            'cooling_slope': round(self.cooling_slope, 4) if self.cooling_slope else None,
            'r2': round(self.r2, 4),
            'rmse': round(self.rmse, 4),
            'mae': round(self.mae, 4),
            'n_observations': self.n_observations,
            'confidence': self.confidence,
            'validation_flags': self.validation_flags,
        }


class ChangePointModel:
    """
    ASHRAE Change-Point Regression Model Implementation.
    
    Supports:
    - 3-Parameter (3P) model: Energy = β₀ + β₁ × max(0, T_balance - T)
    - 4-Parameter (4P) model: Energy = β₀ + β₁ × max(0, T_balance - T) + β₂ × max(0, T - T_balance)
    
    The change-point (T_balance) is found via grid search optimization.
    """
    
    # Confidence thresholds based on R²
    CONFIDENCE_THRESHOLDS = {
        'high': 0.6,    # R² >= 0.6
        'medium': 0.3,  # R² >= 0.3
        'low': 0.0      # R² < 0.3
    }
    
    # Grid search parameters
    T_BALANCE_MIN = 45.0  # Minimum change-point temperature
    T_BALANCE_MAX = 80.0  # Maximum change-point temperature
    T_BALANCE_STEP = 1.0  # Step size for grid search
    
    # Minimum observations required
    MIN_OBSERVATIONS = 30
    
    # Task 5.2: Validation thresholds
    EXTREME_SLOPE = 10_000.0
    EXTREME_T_BALANCE_LOW = 40.0
    EXTREME_T_BALANCE_HIGH = 80.0
    VERY_LOW_R2 = 0.1
    
    def __init__(self):
        """Initialize the ChangePointModel."""
        pass
    
    # ------------------------------------------------------------------
    # Task 5.2: Enhanced model validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_result(result: ModelResult) -> ModelResult:
        """
        Add validation flags to a ModelResult.
        
        Checks:
        - Negative slopes (physically impossible)
        - R² < 0.1 (very poor fit)
        - T_balance outside 40-80°F range
        - Extreme slopes (> 10,000)
        """
        flags = []
        
        if result.heating_slope is not None and result.heating_slope < 0:
            flags.append('negative_heating_slope')
        if result.cooling_slope is not None and result.cooling_slope < 0:
            flags.append('negative_cooling_slope')
        
        if result.r2 < 0.1:
            flags.append('very_poor_fit')
        elif result.r2 < 0.3:
            flags.append('poor_fit')
        
        if result.t_balance < 40.0 or result.t_balance > 80.0:
            flags.append('unusual_t_balance')
        
        max_slope = max(
            abs(result.heating_slope) if result.heating_slope else 0,
            abs(result.cooling_slope) if result.cooling_slope else 0
        )
        if max_slope > 10_000:
            flags.append('extreme_slope')
        
        result.validation_flags = flags
        return result
    
    @staticmethod
    def _fit_3p_at_t(
        temps: np.ndarray, 
        energy: np.ndarray, 
        t_balance: float,
        degree_days_type: str = 'HDD'
    ) -> Tuple[float, float, np.ndarray]:
        """
        Fit 3-parameter model at a specific T_balance.
        
        Args:
            temps: Array of temperatures
            energy: Array of energy values
            t_balance: Change-point temperature
            degree_days_type: 'HDD' for heating, 'CDD' for cooling
            
        Returns:
            Tuple of (intercept, slope, predictions)
        """
        # Calculate degree days based on type
        if degree_days_type == 'HDD':
            # Heating: energy increases when temp < T_balance
            degree_days = np.maximum(0, t_balance - temps)
        else:  # CDD
            # Cooling: energy increases when temp > T_balance
            degree_days = np.maximum(0, temps - t_balance)
        
        # Design matrix: [1, degree_days]
        X = np.column_stack([np.ones(len(temps)), degree_days])
        
        # Solve least squares
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, energy, rcond=None)
            intercept, slope = coeffs[0], coeffs[1]
        except np.linalg.LinAlgError:
            # Fallback to simple mean if lstsq fails
            intercept = np.mean(energy)
            slope = 0.0
        
        # Calculate predictions
        predictions = intercept + slope * degree_days
        
        return intercept, slope, predictions
    
    @staticmethod
    def _fit_4p_at_t(
        temps: np.ndarray, 
        energy: np.ndarray, 
        t_balance: float
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        Fit 4-parameter model at a specific T_balance.
        
        Args:
            temps: Array of temperatures
            energy: Array of energy values
            t_balance: Change-point temperature
            
        Returns:
            Tuple of (intercept, heating_slope, cooling_slope, predictions)
        """
        # Calculate both HDD and CDD
        hdd = np.maximum(0, t_balance - temps)
        cdd = np.maximum(0, temps - t_balance)
        
        # Design matrix: [1, HDD, CDD]
        X = np.column_stack([np.ones(len(temps)), hdd, cdd])
        
        # Solve least squares
        try:
            coeffs, residuals, rank, s = np.linalg.lstsq(X, energy, rcond=None)
            intercept, heating_slope, cooling_slope = coeffs[0], coeffs[1], coeffs[2]
        except np.linalg.LinAlgError:
            # Fallback
            intercept = np.mean(energy)
            heating_slope = 0.0
            cooling_slope = 0.0
        
        # Calculate predictions
        predictions = intercept + heating_slope * hdd + cooling_slope * cdd
        
        return intercept, heating_slope, cooling_slope, predictions
    
    def fit_3p(
        self, 
        temps: np.ndarray, 
        energy: np.ndarray,
        degree_days_type: str = 'HDD',
        t_range: Optional[Tuple[float, float]] = None
    ) -> Optional[ModelResult]:
        """
        Fit a 3-parameter change-point model.
        
        Model: Energy = β₀ + β₁ × max(0, T_balance - T) for HDD
               Energy = β₀ + β₁ × max(0, T - T_balance) for CDD
        
        Args:
            temps: Array of daily average temperatures
            energy: Array of daily energy consumption
            degree_days_type: 'HDD' for heating, 'CDD' for cooling
            t_range: Optional (min, max) range for T_balance search
            
        Returns:
            ModelResult with best fit parameters, or None if insufficient data
        """
        # Validate inputs
        temps = np.asarray(temps, dtype=float)
        energy = np.asarray(energy, dtype=float)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(temps) | np.isnan(energy))
        temps = temps[valid_mask]
        energy = energy[valid_mask]
        
        if len(temps) < self.MIN_OBSERVATIONS:
            print(f"  Warning: Insufficient data ({len(temps)} < {self.MIN_OBSERVATIONS})")
            return None
        
        # Set T_balance search range
        if t_range is None:
            t_min = max(self.T_BALANCE_MIN, temps.min() + 5)
            t_max = min(self.T_BALANCE_MAX, temps.max() - 5)
        else:
            t_min, t_max = t_range
        
        # Grid search for optimal T_balance
        best_r2 = -np.inf
        best_t = None
        best_coeffs = None
        best_predictions = None
        
        for t in np.arange(t_min, t_max + self.T_BALANCE_STEP, self.T_BALANCE_STEP):
            intercept, slope, predictions = self._fit_3p_at_t(
                temps, energy, t, degree_days_type
            )
            
            # Skip if slope is negative (physically impossible)
            if slope < 0:
                continue
            
            # Calculate R²
            r2 = r2_score(energy, predictions)
            
            if r2 > best_r2:
                best_r2 = r2
                best_t = t
                best_coeffs = (intercept, slope)
                best_predictions = predictions
        
        # If no valid fit found
        if best_t is None:
            print(f"  Warning: No valid 3P fit found")
            return None
        
        # Calculate metrics
        residuals = energy - best_predictions
        rmse = np.sqrt(mean_squared_error(energy, best_predictions))
        mae = mean_absolute_error(energy, best_predictions)
        
        # Determine confidence level
        if best_r2 >= self.CONFIDENCE_THRESHOLDS['high']:
            confidence = 'high'
        elif best_r2 >= self.CONFIDENCE_THRESHOLDS['medium']:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        result = ModelResult(
            model_type='3P',
            t_balance=best_t,
            base_load=best_coeffs[0],
            heating_slope=best_coeffs[1] if degree_days_type == 'HDD' else None,
            cooling_slope=best_coeffs[1] if degree_days_type == 'CDD' else None,
            r2=best_r2,
            rmse=rmse,
            mae=mae,
            n_observations=len(temps),
            confidence=confidence,
            predictions=best_predictions,
            residuals=residuals
        )
        return self._validate_result(result)
    
    def fit_4p(
        self, 
        temps: np.ndarray, 
        energy: np.ndarray,
        t_range: Optional[Tuple[float, float]] = None
    ) -> Optional[ModelResult]:
        """
        Fit a 4-parameter change-point model.
        
        Model: Energy = β₀ + β₁ × max(0, T_balance - T) + β₂ × max(0, T - T_balance)
        
        Args:
            temps: Array of daily average temperatures
            energy: Array of daily energy consumption
            t_range: Optional (min, max) range for T_balance search
            
        Returns:
            ModelResult with best fit parameters, or None if insufficient data
        """
        # Validate inputs
        temps = np.asarray(temps, dtype=float)
        energy = np.asarray(energy, dtype=float)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(temps) | np.isnan(energy))
        temps = temps[valid_mask]
        energy = energy[valid_mask]
        
        if len(temps) < self.MIN_OBSERVATIONS:
            print(f"  Warning: Insufficient data ({len(temps)} < {self.MIN_OBSERVATIONS})")
            return None
        
        # Set T_balance search range
        if t_range is None:
            t_min = max(self.T_BALANCE_MIN, temps.min() + 5)
            t_max = min(self.T_BALANCE_MAX, temps.max() - 5)
        else:
            t_min, t_max = t_range
        
        # Grid search for optimal T_balance
        best_r2 = -np.inf
        best_t = None
        best_coeffs = None
        best_predictions = None
        
        for t in np.arange(t_min, t_max + self.T_BALANCE_STEP, self.T_BALANCE_STEP):
            intercept, h_slope, c_slope, predictions = self._fit_4p_at_t(
                temps, energy, t
            )
            
            # Skip if slopes are negative (physically impossible)
            if h_slope < 0 or c_slope < 0:
                continue
            
            # Calculate R²
            r2 = r2_score(energy, predictions)
            
            if r2 > best_r2:
                best_r2 = r2
                best_t = t
                best_coeffs = (intercept, h_slope, c_slope)
                best_predictions = predictions
        
        # If no valid fit found
        if best_t is None:
            print(f"  Warning: No valid 4P fit found")
            return None
        
        # Calculate metrics
        residuals = energy - best_predictions
        rmse = np.sqrt(mean_squared_error(energy, best_predictions))
        mae = mean_absolute_error(energy, best_predictions)
        
        # Determine confidence level
        if best_r2 >= self.CONFIDENCE_THRESHOLDS['high']:
            confidence = 'high'
        elif best_r2 >= self.CONFIDENCE_THRESHOLDS['medium']:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        result = ModelResult(
            model_type='4P',
            t_balance=best_t,
            base_load=best_coeffs[0],
            heating_slope=best_coeffs[1],
            cooling_slope=best_coeffs[2],
            r2=best_r2,
            rmse=rmse,
            mae=mae,
            n_observations=len(temps),
            confidence=confidence,
            predictions=best_predictions,
            residuals=residuals
        )
        return self._validate_result(result)
    
    def fit_best(
        self, 
        temps: np.ndarray, 
        energy: np.ndarray,
        degree_days_type: str = 'HDD',
        prefer_4p: bool = False
    ) -> Optional[ModelResult]:
        """
        Fit the best model (3P or 4P) based on R² improvement.
        
        Args:
            temps: Array of daily average temperatures
            energy: Array of daily energy consumption
            degree_days_type: 'HDD' for heating, 'CDD' for cooling
            prefer_4p: If True, prefer 4P unless 3P is significantly better
            
        Returns:
            ModelResult with best fit, or None if no valid fit
        """
        # Fit both models
        result_3p = self.fit_3p(temps, energy, degree_days_type)
        result_4p = self.fit_4p(temps, energy)
        
        # If only one succeeded, return it
        if result_3p is None and result_4p is None:
            return None
        if result_3p is None:
            return result_4p
        if result_4p is None:
            return result_3p
        
        # Compare models (4P needs at least 0.05 R² improvement to justify complexity)
        r2_improvement_threshold = 0.05
        
        if prefer_4p:
            # Prefer 4P unless 3P is clearly better
            if result_3p.r2 > result_4p.r2 + r2_improvement_threshold:
                return result_3p
            return result_4p
        else:
            # Prefer 3P unless 4P is clearly better (simpler is better)
            if result_4p.r2 > result_3p.r2 + r2_improvement_threshold:
                return result_4p
            return result_3p
    
    def fit_meter(
        self, 
        df: pd.DataFrame,
        utility: str,
        temp_col: str = 'avg_temp',
        energy_col: str = 'energy_sum'
    ) -> Optional[ModelResult]:
        """
        Fit a change-point model for a specific meter's data.
        
        Args:
            df: DataFrame with meter data
            utility: Utility type (HEAT, ELECTRICITY, etc.)
            temp_col: Column name for temperature
            energy_col: Column name for energy
            
        Returns:
            ModelResult or None if fit fails
        """
        # Get degree-days type for this utility
        utility_config = UTILITY_MODELS.get(utility, {})
        degree_days_type = utility_config.get('degree_days', 'HDD')
        
        # Determine if we should try 4P model
        # 4P makes sense for ELECTRICITY which has both heating and cooling components
        prefer_4p = (utility == 'ELECTRICITY')
        
        # Fit the model
        return self.fit_best(
            temps=df[temp_col].values,
            energy=df[energy_col].values,
            degree_days_type=degree_days_type,
            prefer_4p=prefer_4p
        )


def fit_all_meters(
    merged_df: pd.DataFrame,
    progress_callback=None
) -> Dict[Tuple[int, str], ModelResult]:
    """
    Fit change-point models for all building/meter combinations.
    
    Args:
        merged_df: Merged DataFrame from DataProcessor
        progress_callback: Optional callback function(current, total)
        
    Returns:
        Dictionary mapping (building_id, utility) to ModelResult
    """
    model = ChangePointModel()
    results = {}
    
    # Group by building and utility
    groups = merged_df.groupby(['building_id', 'utility'])
    total = len(groups)
    
    print(f"\nFitting change-point models for {total} building/meter combinations...")
    
    for i, ((building_id, utility), group_df) in enumerate(groups):
        if progress_callback:
            progress_callback(i + 1, total)
        
        # Fit the model
        result = model.fit_meter(group_df, utility)
        
        if result is not None:
            results[(building_id, utility)] = result
        
        # Progress update every 50 meters
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  Processed {i + 1}/{total} meters ({len(results)} successful fits)")
    
    print(f"\nCompleted: {len(results)}/{total} successful model fits")
    
    # Summary statistics
    if results:
        r2_values = [r.r2 for r in results.values()]
        confidence_counts = {}
        for r in results.values():
            confidence_counts[r.confidence] = confidence_counts.get(r.confidence, 0) + 1
        
        print(f"  R² range: {min(r2_values):.3f} - {max(r2_values):.3f}")
        print(f"  R² mean: {np.mean(r2_values):.3f}")
        print(f"  Confidence distribution: {confidence_counts}")
    
    return results


def generate_vcurve_data(
    df: pd.DataFrame,
    result: ModelResult,
    temp_col: str = 'avg_temp',
    energy_col: str = 'energy_sum'
) -> Dict[str, Any]:
    """
    Generate V-curve visualization data for a model result.
    
    Args:
        df: DataFrame with meter data
        result: ModelResult from model fitting
        temp_col: Column name for temperature
        energy_col: Column name for energy
        
    Returns:
        Dictionary with V-curve data for visualization
    """
    temps = df[temp_col].values
    energy = df[energy_col].values
    
    # Generate fitted line data points
    temp_range = np.linspace(temps.min(), temps.max(), 100)
    
    if result.model_type == '3P':
        if result.heating_slope is not None:
            # Heating model (HDD)
            fitted = result.base_load + result.heating_slope * np.maximum(0, result.t_balance - temp_range)
        else:
            # Cooling model (CDD)
            fitted = result.base_load + result.cooling_slope * np.maximum(0, temp_range - result.t_balance)
    else:
        # 4P model
        hdd = np.maximum(0, result.t_balance - temp_range)
        cdd = np.maximum(0, temp_range - result.t_balance)
        fitted = result.base_load + result.heating_slope * hdd + result.cooling_slope * cdd
    
    return {
        'actual': {
            'temperatures': temps.tolist(),
            'energy': energy.tolist()
        },
        'fitted': {
            'temperatures': temp_range.tolist(),
            'energy': fitted.tolist()
        },
        'change_point': {
            'temperature': result.t_balance,
            'energy': result.base_load
        },
        'model_info': result.to_dict()
    }


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    # Generate sample data for testing
    np.random.seed(42)
    n = 365
    
    # Simulate heating-dominant building
    temps = np.random.uniform(20, 90, n)
    base_load = 100
    heating_slope = 5
    t_balance = 65
    
    hdd = np.maximum(0, t_balance - temps)
    energy = base_load + heating_slope * hdd + np.random.normal(0, 10, n)
    
    print("Testing 3P Model (Heating):")
    print("  True parameters: base_load=100, heating_slope=5, t_balance=65")
    
    model = ChangePointModel()
    result = model.fit_3p(temps, energy, 'HDD')
    
    if result:
        print(f"  Fitted parameters: base_load={result.base_load:.2f}, "
              f"heating_slope={result.heating_slope:.2f}, t_balance={result.t_balance:.2f}")
        print(f"  R²: {result.r2:.4f}, RMSE: {result.rmse:.2f}, Confidence: {result.confidence}")
    
    print("\nTesting 4P Model:")
    # Simulate building with both heating and cooling
    cooling_slope = 3
    cdd = np.maximum(0, temps - t_balance)
    energy_4p = base_load + heating_slope * hdd + cooling_slope * cdd + np.random.normal(0, 10, n)
    
    result_4p = model.fit_4p(temps, energy_4p)
    
    if result_4p:
        print(f"  Fitted parameters: base_load={result_4p.base_load:.2f}, "
              f"heating_slope={result_4p.heating_slope:.2f}, "
              f"cooling_slope={result_4p.cooling_slope:.2f}, t_balance={result_4p.t_balance:.2f}")
        print(f"  R²: {result_4p.r2:.4f}, RMSE: {result_4p.rmse:.2f}, Confidence: {result_4p.confidence}")
