"""
Building Aggregator Module for Eco-Pulse
Handles multi-meter building aggregation and efficiency scoring.

Production-ready with:
- Percentile-based normalization (Task 4.1)
- Size-category normalization (Task 4.2)
- Graceful NULL handling (Task 4.3)
- Data quality flags (Task 5.1)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .change_point import ModelResult, UTILITY_MODELS


@dataclass
class MeterScore:
    """Score for a single meter."""
    building_id: int
    utility: str
    degree_days_type: str
    slope: float
    t_balance: float
    base_load: float
    total_energy: float
    r2: float
    confidence: str
    # Task 5.1: quality flags
    data_quality_flags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'building_id': self.building_id,
            'utility': self.utility,
            'degree_days_type': self.degree_days_type,
            'slope': round(self.slope, 4),
            't_balance': round(self.t_balance, 2),
            'base_load': round(self.base_load, 4),
            'total_energy': round(self.total_energy, 2),
            'r2': round(self.r2, 4),
            'confidence': self.confidence,
            'data_quality_flags': self.data_quality_flags,
        }


@dataclass
class BuildingScore:
    """Aggregated efficiency score for a building."""
    building_id: int
    building_name: str
    gross_area: float
    building_type: str = 'regular'
    
    heating_slope: Optional[float] = None
    cooling_slope: Optional[float] = None
    composite_slope: Optional[float] = None
    
    total_energy: float = 0.0
    heating_energy: float = 0.0
    cooling_energy: float = 0.0
    eui: float = 0.0
    
    heating_efficiency_score: Optional[float] = None
    cooling_efficiency_score: Optional[float] = None
    overall_efficiency_score: Optional[float] = None
    efficiency_rank: Optional[int] = None
    
    avg_r2: float = 0.0
    avg_confidence: str = 'low'
    num_meters: int = 0
    
    # Task 4.2: size category
    size_category: str = 'medium'
    
    # Task 5.1: data quality flags
    data_quality_flags: List[str] = field(default_factory=list)
    has_outliers: bool = False
    unit_converted: bool = False
    
    # Data coverage: 'full', 'partial', or 'electricity_only'
    data_coverage: str = 'full'
    
    meter_scores: List[MeterScore] = field(default_factory=list)
    
    def to_dict(self, include_meters: bool = False) -> Dict[str, Any]:
        result = {
            'building_id': self.building_id,
            'building_name': self.building_name,
            'gross_area': round(self.gross_area, 2),
            'building_type': self.building_type,
            'heating_slope': round(self.heating_slope, 4) if self.heating_slope is not None else None,
            'cooling_slope': round(self.cooling_slope, 4) if self.cooling_slope is not None else None,
            'composite_slope': round(self.composite_slope, 4) if self.composite_slope is not None else None,
            'total_energy': round(self.total_energy, 2),
            'heating_energy': round(self.heating_energy, 2),
            'cooling_energy': round(self.cooling_energy, 2),
            'eui': round(self.eui, 4),
            'heating_efficiency_score': round(self.heating_efficiency_score, 2) if self.heating_efficiency_score is not None else None,
            'cooling_efficiency_score': round(self.cooling_efficiency_score, 2) if self.cooling_efficiency_score is not None else None,
            'overall_efficiency_score': round(self.overall_efficiency_score, 2) if self.overall_efficiency_score is not None else None,
            'efficiency_rank': self.efficiency_rank,
            'avg_r2': round(self.avg_r2, 4),
            'avg_confidence': self.avg_confidence,
            'num_meters': self.num_meters,
            'size_category': self.size_category,
            'data_quality_flags': self.data_quality_flags,
            'has_outliers': self.has_outliers,
            'unit_converted': self.unit_converted,
            'data_coverage': self.data_coverage,
        }
        if include_meters:
            result['meters'] = [m.to_dict() for m in self.meter_scores]
        return result


# ====================================================================
#  Size-category thresholds (Task 4.2)
# ====================================================================
SIZE_CATEGORIES = {
    'small':  (0, 50_000),
    'medium': (50_000, 200_000),
    'large':  (200_000, float('inf')),
}


class BuildingAggregator:
    """
    Aggregates meter-level change-point results to building-level efficiency scores.
    """
    
    CONFIDENCE_SCORES = {'high': 1.0, 'medium': 0.5, 'low': 0.25}
    
    # Task 5.1 thresholds
    EXTREME_SLOPE_THRESHOLD = 100.0   # slopes above this are flagged
    LOW_R2_THRESHOLD = 0.3
    VERY_LOW_R2_THRESHOLD = 0.1
    EXTREME_T_BALANCE_LOW = 40.0
    EXTREME_T_BALANCE_HIGH = 80.0
    
    def __init__(self, building_metadata: pd.DataFrame):
        self.building_metadata = building_metadata.set_index('building_id').to_dict('index')
        self.building_scores: Dict[int, BuildingScore] = {}
        self.rankings: List[int] = []
        # Task 4.1: will be populated during aggregate_all_buildings
        self._slope_percentiles: Dict[str, Dict[str, float]] = {}
    
    # ----------------------------------------------------------------
    #  Task 4.2: get size category
    # ----------------------------------------------------------------
    @staticmethod
    def _size_category(gross_area: float) -> str:
        for cat, (lo, hi) in SIZE_CATEGORIES.items():
            if lo <= gross_area < hi:
                return cat
        return 'medium'
    
    # ----------------------------------------------------------------
    #  Helpers
    # ----------------------------------------------------------------
    def _get_slope_for_meter(self, result: ModelResult, utility: str) -> Tuple[float, str]:
        utility_config = UTILITY_MODELS.get(utility, {})
        degree_days_type = utility_config.get('degree_days', 'HDD')
        if degree_days_type == 'HDD':
            slope = result.heating_slope if result.heating_slope else 0.0
        else:
            slope = result.cooling_slope if result.cooling_slope else 0.0
        return slope, degree_days_type
    
    # ----------------------------------------------------------------
    #  Task 5.1: generate quality flags for a meter
    # ----------------------------------------------------------------
    def _meter_quality_flags(self, result: ModelResult, slope: float) -> List[str]:
        flags = []
        if result.r2 < self.VERY_LOW_R2_THRESHOLD:
            flags.append('very_low_r2')
        elif result.r2 < self.LOW_R2_THRESHOLD:
            flags.append('low_r2')
        if abs(slope) > self.EXTREME_SLOPE_THRESHOLD:
            flags.append('extreme_slope')
        if result.t_balance < self.EXTREME_T_BALANCE_LOW or result.t_balance > self.EXTREME_T_BALANCE_HIGH:
            flags.append('unusual_t_balance')
        if result.heating_slope is not None and result.heating_slope < 0:
            flags.append('negative_slope')
        if result.cooling_slope is not None and result.cooling_slope < 0:
            flags.append('negative_slope')
        return flags
    
    # ================================================================
    #  Aggregate a single building
    # ================================================================
    def aggregate_building(
        self,
        building_id: int,
        meter_results: Dict[str, ModelResult],
        energy_totals: Dict[str, float]
    ) -> BuildingScore:
        metadata = self.building_metadata.get(building_id, {})
        building_name = metadata.get('building_name', f'Building {building_id}')
        gross_area = metadata.get('gross_area', 1.0)
        building_type = metadata.get('building_type', 'regular')
        size_cat = self._size_category(gross_area)
        
        # Collect meters
        meter_scores = []
        heating_meters = []
        cooling_meters = []
        total_energy = 0.0
        heating_energy = 0.0
        cooling_energy = 0.0
        building_flags: List[str] = []
        has_outliers = False
        unit_converted = False
        
        for utility, result in meter_results.items():
            slope, dd_type = self._get_slope_for_meter(result, utility)
            total_meter_energy = energy_totals.get(utility, 0.0)
            
            # quality flags
            meter_flags = self._meter_quality_flags(result, slope)
            if 'extreme_slope' in meter_flags:
                has_outliers = True
            
            # Check if this was a STEAM utility (unit_converted)
            if utility == 'STEAM':
                unit_converted = True
                meter_flags.append('unit_converted')
            
            ms = MeterScore(
                building_id=building_id,
                utility=utility,
                degree_days_type=dd_type,
                slope=slope,
                t_balance=result.t_balance,
                base_load=result.base_load,
                total_energy=total_meter_energy,
                r2=result.r2,
                confidence=result.confidence,
                data_quality_flags=meter_flags,
            )
            meter_scores.append(ms)
            total_energy += total_meter_energy
            
            if dd_type == 'HDD':
                heating_meters.append((slope, total_meter_energy, result.r2, result.confidence))
                heating_energy += total_meter_energy
            else:
                cooling_meters.append((slope, total_meter_energy, result.r2, result.confidence))
                cooling_energy += total_meter_energy
        
        # Weighted average slopes
        heating_slope = self._weighted_avg_slope(heating_meters)
        cooling_slope = self._weighted_avg_slope(cooling_meters)
        
        # Composite slope
        composite_slope = None
        if heating_slope is not None or cooling_slope is not None:
            total_w = 0.0
            w_sum = 0.0
            if heating_slope is not None and heating_energy > 0:
                w_sum += heating_slope * heating_energy
                total_w += heating_energy
            if cooling_slope is not None and cooling_energy > 0:
                w_sum += cooling_slope * cooling_energy
                total_w += cooling_energy
            if total_w > 0:
                composite_slope = w_sum / total_w
        
        # EUI
        eui = total_energy / gross_area if gross_area > 0 else 0.0
        
        # Average R² and confidence
        all_r2 = [m.r2 for m in meter_scores]
        avg_r2 = float(np.mean(all_r2)) if all_r2 else 0.0
        confidence_sum = sum(self.CONFIDENCE_SCORES.get(m.confidence, 0.25) for m in meter_scores)
        avg_conf_score = confidence_sum / len(meter_scores) if meter_scores else 0.0
        if avg_conf_score >= 0.8:
            avg_confidence = 'high'
        elif avg_conf_score >= 0.4:
            avg_confidence = 'medium'
        else:
            avg_confidence = 'low'
        
        # Build building-level flags
        if has_outliers:
            building_flags.append('has_outliers')
        if unit_converted:
            building_flags.append('unit_converted')
        if avg_r2 < self.LOW_R2_THRESHOLD:
            building_flags.append('low_data_quality')
        
        # Data coverage: does this building have both heating and cooling data?
        has_hdd = any(m.degree_days_type == 'HDD' for m in meter_scores)
        has_cdd = any(m.degree_days_type == 'CDD' for m in meter_scores)
        utility_set = {m.utility for m in meter_scores}
        if has_hdd and has_cdd:
            data_coverage = 'full'
        elif utility_set == {'ELECTRICITY'}:
            data_coverage = 'electricity_only'
        else:
            data_coverage = 'partial'
        
        if data_coverage != 'full':
            building_flags.append('partial_data')
        
        # NOTE: efficiency scores are set to None here;
        # they will be assigned in aggregate_all_buildings after percentile calculation
        return BuildingScore(
            building_id=building_id,
            building_name=building_name,
            gross_area=gross_area,
            building_type=building_type,
            heating_slope=heating_slope,
            cooling_slope=cooling_slope,
            composite_slope=composite_slope,
            total_energy=total_energy,
            heating_energy=heating_energy,
            cooling_energy=cooling_energy,
            eui=eui,
            heating_efficiency_score=None,
            cooling_efficiency_score=None,
            overall_efficiency_score=None,
            avg_r2=avg_r2,
            avg_confidence=avg_confidence,
            num_meters=len(meter_scores),
            size_category=size_cat,
            data_quality_flags=building_flags,
            has_outliers=has_outliers,
            unit_converted=unit_converted,
            data_coverage=data_coverage,
            meter_scores=meter_scores,
        )
    
    @staticmethod
    def _weighted_avg_slope(meters: list) -> Optional[float]:
        """Calculate energy-weighted average slope."""
        if not meters:
            return None
        total_e = sum(m[1] for m in meters)
        if total_e <= 0:
            return None
        return sum(m[0] * m[1] for m in meters) / total_e
    
    # ================================================================
    #  Aggregate ALL buildings  (includes percentile normalization)
    # ================================================================
    def aggregate_all_buildings(
        self,
        model_results: Dict[Tuple[int, str], ModelResult],
        merged_df: pd.DataFrame
    ) -> Dict[int, BuildingScore]:
        print("\nAggregating building scores...")
        
        # Group model results by building
        building_meters: Dict[int, Dict[str, ModelResult]] = {}
        for (building_id, utility), result in model_results.items():
            building_meters.setdefault(building_id, {})[utility] = result
        
        energy_totals = merged_df.groupby(['building_id', 'utility'])['energy_sum'].sum().to_dict()
        
        # First pass: aggregate each building (slopes, EUI, etc.)
        building_scores: Dict[int, BuildingScore] = {}
        for building_id, meter_results in building_meters.items():
            building_energy = {
                u: energy_totals.get((building_id, u), 0.0) for u in meter_results
            }
            score = self.aggregate_building(building_id, meter_results, building_energy)
            building_scores[building_id] = score
        
        print(f"  Aggregated {len(building_scores)} buildings")
        
        # ---- Task 4.1: Percentile-based normalization ----
        self._apply_percentile_normalization(building_scores)
        
        # ---- Task 4.3: Handle NULL scores ----
        self._fill_null_scores(building_scores)
        
        # ---- Final clamp (safety against floating-point drift) ----
        for s in building_scores.values():
            if s.overall_efficiency_score is not None:
                s.overall_efficiency_score = float(np.clip(s.overall_efficiency_score, 0.0, 100.0))
            if s.heating_efficiency_score is not None:
                s.heating_efficiency_score = float(np.clip(s.heating_efficiency_score, 0.0, 100.0))
            if s.cooling_efficiency_score is not None:
                s.cooling_efficiency_score = float(np.clip(s.cooling_efficiency_score, 0.0, 100.0))
        
        # ---- Rankings ----
        self._calculate_rankings(building_scores)
        
        self.building_scores = building_scores
        return building_scores
    
    # ================================================================
    #  Task 4.1 + 4.2: Percentile-based normalization
    # ================================================================
    def _apply_percentile_normalization(self, scores: Dict[int, BuildingScore]):
        """
        Normalise slopes to 0-100 scores using percentiles.
        
        - 10th percentile slope = "best" → score 100
        - 90th percentile slope = "worst" → score 0
        - Clamped to [0, 100]
        
        Task 4.2: percentiles are calculated *per size category*.
        """
        # Collect slopes by (degree_days_type, size_category)
        heat_slopes: Dict[str, List[float]] = {}
        cool_slopes: Dict[str, List[float]] = {}
        
        for s in scores.values():
            cat = s.size_category
            if s.heating_slope is not None:
                heat_slopes.setdefault(cat, []).append(s.heating_slope)
            if s.cooling_slope is not None:
                cool_slopes.setdefault(cat, []).append(s.cooling_slope)
        
        # Also build "all" bucket as fallback
        all_heat = [s.heating_slope for s in scores.values() if s.heating_slope is not None]
        all_cool = [s.cooling_slope for s in scores.values() if s.cooling_slope is not None]
        heat_slopes['_all'] = all_heat
        cool_slopes['_all'] = all_cool
        
        # Precompute p10/p90 for each group
        heat_p = {}
        for k, v in heat_slopes.items():
            if len(v) >= 5:
                heat_p[k] = (float(np.percentile(v, 10)), float(np.percentile(v, 90)))
            elif len(v) > 0:
                heat_p[k] = (min(v), max(v))
        
        cool_p = {}
        for k, v in cool_slopes.items():
            if len(v) >= 5:
                cool_p[k] = (float(np.percentile(v, 10)), float(np.percentile(v, 90)))
            elif len(v) > 0:
                cool_p[k] = (min(v), max(v))
        
        # Apply normalization
        for s in scores.values():
            cat = s.size_category
            
            # Heating score
            if s.heating_slope is not None:
                p10, p90 = heat_p.get(cat, heat_p.get('_all', (0.5, 10.0)))
                s.heating_efficiency_score = self._slope_to_score(s.heating_slope, p10, p90)
            
            # Cooling score
            if s.cooling_slope is not None:
                p10, p90 = cool_p.get(cat, cool_p.get('_all', (0.5, 8.0)))
                s.cooling_efficiency_score = self._slope_to_score(s.cooling_slope, p10, p90)
            
            # Overall: energy-weighted average
            s.overall_efficiency_score = self._overall_score(s)
        
        # Store for reference
        self._slope_percentiles = {'heat': heat_p, 'cool': cool_p}
        
        # Stats
        valid = [s.overall_efficiency_score for s in scores.values() if s.overall_efficiency_score is not None]
        if valid:
            print(f"  Percentile normalization applied")
            print(f"  Score range: {min(valid):.1f} - {max(valid):.1f}  (mean {np.mean(valid):.1f})")
    
    @staticmethod
    def _slope_to_score(slope: float, p10: float, p90: float) -> float:
        """
        Convert slope to 0–100 score.  Lower slope = higher score.
        p10 (best) → 100,  p90 (worst) → 0.
        """
        if p90 <= p10:
            return 50.0  # degenerate case, return median
        raw = 100.0 * (1.0 - (slope - p10) / (p90 - p10))
        return float(np.clip(raw, 0.0, 100.0))
    
    @staticmethod
    def _overall_score(s: 'BuildingScore') -> Optional[float]:
        """Energy-weighted combination of heating and cooling scores."""
        w_sum = 0.0
        total_w = 0.0
        if s.heating_efficiency_score is not None and s.heating_energy > 0:
            w_sum += s.heating_efficiency_score * s.heating_energy
            total_w += s.heating_energy
        if s.cooling_efficiency_score is not None and s.cooling_energy > 0:
            w_sum += s.cooling_efficiency_score * s.cooling_energy
            total_w += s.cooling_energy
        if total_w > 0:
            return w_sum / total_w
        # Fallback: whichever is available
        if s.heating_efficiency_score is not None:
            return s.heating_efficiency_score
        if s.cooling_efficiency_score is not None:
            return s.cooling_efficiency_score
        return None
    
    # ================================================================
    #  Task 4.3: fill NULL scores with median + low-confidence flag
    # ================================================================
    def _fill_null_scores(self, scores: Dict[int, BuildingScore]):
        """Ensure no building has a None overall_efficiency_score."""
        valid = [
            s.overall_efficiency_score
            for s in scores.values()
            if s.overall_efficiency_score is not None
        ]
        median_score = float(np.median(valid)) if valid else 50.0
        
        null_count = 0
        for s in scores.values():
            if s.overall_efficiency_score is None:
                s.overall_efficiency_score = median_score
                s.avg_confidence = 'low'
                if 'null_score_imputed' not in s.data_quality_flags:
                    s.data_quality_flags.append('null_score_imputed')
                null_count += 1
        
        if null_count > 0:
            print(f"  Imputed {null_count} NULL scores with median ({median_score:.1f})")
    
    # ================================================================
    #  Rankings
    # ================================================================
    def _calculate_rankings(self, building_scores: Dict[int, BuildingScore]):
        ranked = sorted(
            building_scores.items(),
            key=lambda x: x[1].overall_efficiency_score if x[1].overall_efficiency_score is not None else 0,
            reverse=True
        )
        for rank, (bid, score) in enumerate(ranked, 1):
            score.efficiency_rank = rank
        self.rankings = [bid for bid, _ in ranked]
        
        print(f"  Rankings calculated")
        if ranked:
            top = ranked[0][1]
            bottom = ranked[-1][1]
            print(f"  Top: {top.building_name} (score: {top.overall_efficiency_score:.1f})")
            print(f"  Bottom: {bottom.building_name} (score: {bottom.overall_efficiency_score:.1f})")
    
    # ================================================================
    #  Public query methods
    # ================================================================
    # Minimum gross area to include in rankings (excludes sheds/tiny structures)
    MIN_RANKING_AREA = 1_000  # sq ft
    
    def get_rankings(
        self,
        limit: Optional[int] = None,
        sort_by: str = 'overall_efficiency_score',
        ascending: bool = False,
        exclude_special: bool = False,
        min_confidence: Optional[str] = None,
        exclude_outliers: bool = False,
        min_area: Optional[float] = None,
    ) -> List[BuildingScore]:
        scores = list(self.building_scores.values())
        
        # Exclude tiny buildings (sheds, storage, kiosks) by default
        area_threshold = min_area if min_area is not None else self.MIN_RANKING_AREA
        scores = [s for s in scores if s.gross_area >= area_threshold]
        
        # Task 6.2 filters
        if exclude_special:
            scores = [s for s in scores if s.building_type == 'regular']
        if exclude_outliers:
            scores = [s for s in scores if not s.has_outliers]
        if min_confidence:
            order = {'high': 3, 'medium': 2, 'low': 1}
            min_level = order.get(min_confidence.lower(), 0)
            scores = [s for s in scores if order.get(s.avg_confidence, 0) >= min_level]
        
        def get_sort_key(score):
            val = getattr(score, sort_by, None)
            return val if val is not None else (0 if ascending else float('inf'))
        
        scores.sort(key=get_sort_key, reverse=not ascending)
        if limit:
            scores = scores[:limit]
        return scores
    
    def get_building_score(self, building_id: int) -> Optional[BuildingScore]:
        return self.building_scores.get(building_id)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        if not self.building_scores:
            return {}
        scores = list(self.building_scores.values())
        eff = [s.overall_efficiency_score for s in scores if s.overall_efficiency_score is not None]
        heat = [s.heating_efficiency_score for s in scores if s.heating_efficiency_score is not None]
        cool = [s.cooling_efficiency_score for s in scores if s.cooling_efficiency_score is not None]
        euis = [s.eui for s in scores]
        return {
            'total_buildings': len(scores),
            'overall_efficiency': {
                'mean': float(np.mean(eff)) if eff else None,
                'median': float(np.median(eff)) if eff else None,
                'std': float(np.std(eff)) if eff else None,
                'min': float(min(eff)) if eff else None,
                'max': float(max(eff)) if eff else None,
            },
            'heating_efficiency': {
                'mean': float(np.mean(heat)) if heat else None,
                'count': len(heat),
            },
            'cooling_efficiency': {
                'mean': float(np.mean(cool)) if cool else None,
                'count': len(cool),
            },
            'eui': {
                'mean': float(np.mean(euis)) if euis else None,
                'median': float(np.median(euis)) if euis else None,
                'min': float(min(euis)) if euis else None,
                'max': float(max(euis)) if euis else None,
            },
            'confidence_distribution': {
                'high': sum(1 for s in scores if s.avg_confidence == 'high'),
                'medium': sum(1 for s in scores if s.avg_confidence == 'medium'),
                'low': sum(1 for s in scores if s.avg_confidence == 'low'),
            },
            'quality_flag_counts': self._quality_flag_summary(scores),
        }
    
    @staticmethod
    def _quality_flag_summary(scores: List[BuildingScore]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in scores:
            for flag in s.data_quality_flags:
                counts[flag] = counts.get(flag, 0) + 1
        return counts
    
    def to_dataframe(self) -> pd.DataFrame:
        records = [score.to_dict(include_meters=False) for score in self.building_scores.values()]
        return pd.DataFrame(records)
    
    # ================================================================
    #  Action Plans -- retrofit recommendations with savings estimates
    # ================================================================
    
    # Configurable constants
    COST_PER_KWH = 0.10       # $/kWh (typical Ohio State utility rate)
    ANNUAL_HDD = 5529.0       # From 2025 weather data
    ANNUAL_CDD = 1025.0       # From 2025 weather data
    MAX_SAVINGS_PCT = 0.40    # Cap total savings at 40% of est. annual consumption
    
    # Diagnosis thresholds (multiplier above median to trigger recommendation)
    SLOPE_CRITICAL_MULT = 3.0   # >3x median = critical
    SLOPE_HIGH_MULT = 2.0       # >2x median = high
    SLOPE_MEDIUM_MULT = 1.3     # >1.3x median = medium
    
    def _get_median_slopes(self) -> Dict[str, Dict[str, float]]:
        """
        Compute median *per-sqft* heating slope, cooling slope, and base load
        per size category from current building_scores.
        
        All values are normalised by gross_area so that buildings of different
        sizes within the same category are compared fairly.
        
        Returns: {'small': {'heat': X, 'cool': Y, 'base': Z}, ...}
                 where X = median kWh/DD/sqft, Z = median kWh/day/sqft
        """
        by_cat: Dict[str, Dict[str, list]] = {}
        
        for s in self.building_scores.values():
            if s.building_type != 'regular' or s.gross_area < self.MIN_RANKING_AREA:
                continue
            cat = s.size_category
            area = s.gross_area
            by_cat.setdefault(cat, {'heat': [], 'cool': [], 'base': []})
            if s.heating_slope is not None:
                by_cat[cat]['heat'].append(s.heating_slope / area)
            if s.cooling_slope is not None:
                by_cat[cat]['cool'].append(s.cooling_slope / area)
            # base load: total building daily base / area
            total_base = sum(m.base_load for m in s.meter_scores if m.base_load > 0)
            if total_base > 0:
                by_cat[cat]['base'].append(total_base / area)
        
        result = {}
        for cat, data in by_cat.items():
            result[cat] = {
                'heat': float(np.median(data['heat'])) if data['heat'] else 0.0,
                'cool': float(np.median(data['cool'])) if data['cool'] else 0.0,
                'base': float(np.median(data['base'])) if data['base'] else 0.0,
            }
        return result
    
    def generate_action_plan(self, building_id: int) -> Optional[Dict[str, Any]]:
        """
        Generate a physics-based action plan for a single building.
        
        Compares the building's per-sqft slopes and base load to the
        median for its size category. Savings are then scaled back up
        by the building's area so the dollar figures represent the
        actual building.
        """
        score = self.building_scores.get(building_id)
        if score is None:
            return None
        
        area = score.gross_area
        if area <= 0:
            return None
        
        medians = self._get_median_slopes()  # all values are per-sqft
        cat_med = medians.get(score.size_category,
                              medians.get('medium', {'heat': 0, 'cool': 0, 'base': 0}))
        
        med_heat_psf = cat_med['heat']   # kWh / DD / sqft
        med_cool_psf = cat_med['cool']   # kWh / DD / sqft
        med_base_psf = cat_med['base']   # kWh / day / sqft
        
        recommendations: List[Dict[str, Any]] = []
        total_kwh_savings = 0.0
        total_cost_savings = 0.0
        
        # --- Heating / insulation analysis (per-sqft comparison) ---
        if score.heating_slope is not None and med_heat_psf > 0:
            heat_psf = score.heating_slope / area
            ratio = heat_psf / med_heat_psf
            if ratio > self.SLOPE_MEDIUM_MULT:
                reduction_psf = heat_psf - med_heat_psf
                kwh_saved = reduction_psf * area * self.ANNUAL_HDD
                cost_saved = kwh_saved * self.COST_PER_KWH
                
                if ratio > self.SLOPE_CRITICAL_MULT:
                    priority = 'critical'
                elif ratio > self.SLOPE_HIGH_MULT:
                    priority = 'high'
                else:
                    priority = 'medium'
                
                recommendations.append({
                    'type': 'insulation_upgrade',
                    'title': 'Insulation / Envelope Upgrade',
                    'description': (
                        f'Heating intensity is {ratio:.1f}x the median for {score.size_category}-sized buildings. '
                        f'This building uses {reduction_psf:.4f} extra kWh/sqft per Heating Degree Day. '
                        f'Recommended actions: upgrade wall/roof insulation, seal air leaks, replace windows.'
                    ),
                    'priority': priority,
                    'current_slope': round(score.heating_slope, 2),
                    'target_slope': round(med_heat_psf * area, 2),
                    'estimated_kwh_savings': round(kwh_saved, 0),
                    'estimated_cost_savings': round(cost_saved, 0),
                })
                total_kwh_savings += kwh_saved
                total_cost_savings += cost_saved
        
        # --- Cooling / HVAC analysis (per-sqft comparison) ---
        if score.cooling_slope is not None and med_cool_psf > 0:
            cool_psf = score.cooling_slope / area
            ratio = cool_psf / med_cool_psf
            if ratio > self.SLOPE_MEDIUM_MULT:
                reduction_psf = cool_psf - med_cool_psf
                kwh_saved = reduction_psf * area * self.ANNUAL_CDD
                cost_saved = kwh_saved * self.COST_PER_KWH
                
                if ratio > self.SLOPE_CRITICAL_MULT:
                    priority = 'critical'
                elif ratio > self.SLOPE_HIGH_MULT:
                    priority = 'high'
                else:
                    priority = 'medium'
                
                recommendations.append({
                    'type': 'hvac_upgrade',
                    'title': 'HVAC System Upgrade',
                    'description': (
                        f'Cooling intensity is {ratio:.1f}x the median for {score.size_category}-sized buildings. '
                        f'This building uses {reduction_psf:.4f} extra kWh/sqft per Cooling Degree Day. '
                        f'Recommended actions: replace aging chillers/AHUs, upgrade controls, add economizers.'
                    ),
                    'priority': priority,
                    'current_slope': round(score.cooling_slope, 2),
                    'target_slope': round(med_cool_psf * area, 2),
                    'estimated_kwh_savings': round(kwh_saved, 0),
                    'estimated_cost_savings': round(cost_saved, 0),
                })
                total_kwh_savings += kwh_saved
                total_cost_savings += cost_saved
        
        # --- Base load analysis (per-sqft comparison) ---
        total_base = float(sum(m.base_load for m in score.meter_scores)) if score.meter_scores else 0.0
        if med_base_psf > 0 and total_base > 0:
            base_psf = total_base / area
            base_ratio = base_psf / med_base_psf
            if base_ratio > self.SLOPE_MEDIUM_MULT:
                reduction_psf = base_psf - med_base_psf
                kwh_saved = reduction_psf * area * 365
                cost_saved = kwh_saved * self.COST_PER_KWH
                
                if base_ratio > self.SLOPE_CRITICAL_MULT:
                    priority = 'critical'
                elif base_ratio > self.SLOPE_HIGH_MULT:
                    priority = 'high'
                else:
                    priority = 'medium'
                
                recommendations.append({
                    'type': 'base_load_reduction',
                    'title': 'Base Load Reduction',
                    'description': (
                        f'Weather-independent base load is {base_ratio:.1f}x the median per sqft. '
                        f'The building consumes {reduction_psf:.4f} extra kWh/sqft/day regardless of weather. '
                        f'Recommended actions: LED lighting retrofit, optimize schedules, reduce plug loads, '
                        f'fix equipment running 24/7.'
                    ),
                    'priority': priority,
                    'current_base_load': round(total_base, 2),
                    'target_base_load': round(med_base_psf * area, 2),
                    'estimated_kwh_savings': round(kwh_saved, 0),
                    'estimated_cost_savings': round(cost_saved, 0),
                })
                total_kwh_savings += kwh_saved
                total_cost_savings += cost_saved
        
        # --- If both heating and cooling are bad, upgrade to comprehensive ---
        has_insulation = any(r['type'] == 'insulation_upgrade' for r in recommendations)
        has_hvac = any(r['type'] == 'hvac_upgrade' for r in recommendations)
        if has_insulation and has_hvac:
            recommendations.append({
                'type': 'comprehensive_retrofit',
                'title': 'Comprehensive Energy Retrofit',
                'description': (
                    'Both heating and cooling systems are significantly underperforming. '
                    'A comprehensive energy audit and retrofit is recommended to address '
                    'envelope, HVAC, controls, and operational issues together for maximum savings.'
                ),
                'priority': 'critical',
                'estimated_kwh_savings': 0,   # already counted above
                'estimated_cost_savings': 0,
            })
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2}
        recommendations.sort(key=lambda r: priority_order.get(r.get('priority', 'medium'), 9))
        
        # Cap total savings at MAX_SAVINGS_PCT of estimated annual consumption
        # to avoid physically impossible recommendations.
        est_annual_kwh = score.eui * score.gross_area  # eui is annualized
        max_kwh = est_annual_kwh * self.MAX_SAVINGS_PCT if est_annual_kwh > 0 else float('inf')
        if total_kwh_savings > max_kwh and max_kwh > 0:
            scale = max_kwh / total_kwh_savings
            total_kwh_savings = max_kwh
            total_cost_savings = total_kwh_savings * self.COST_PER_KWH
            # Scale individual recommendation estimates proportionally
            for rec in recommendations:
                rec['estimated_kwh_savings'] = round(rec['estimated_kwh_savings'] * scale, 0)
                rec['estimated_cost_savings'] = round(rec['estimated_cost_savings'] * scale, 0)
        
        return {
            'building_id': score.building_id,
            'building_name': score.building_name,
            'gross_area': round(score.gross_area, 0),
            'size_category': score.size_category,
            'current_efficiency_score': round(score.overall_efficiency_score, 1) if score.overall_efficiency_score is not None else None,
            'current_eui': round(score.eui, 2),
            'heating_slope': round(score.heating_slope, 2) if score.heating_slope is not None else None,
            'cooling_slope': round(score.cooling_slope, 2) if score.cooling_slope is not None else None,
            'recommendations': recommendations,
            'total_estimated_kwh_savings': round(total_kwh_savings, 0),
            'total_estimated_cost_savings': round(total_cost_savings, 0),
            'methodology_note': (
                f'Savings estimated by comparing per-sqft energy intensity to the '
                f'median for {score.size_category}-sized buildings, then scaling by '
                f'building area ({score.gross_area:,.0f} sqft). '
                f'Capped at {self.MAX_SAVINGS_PCT*100:.0f}% of estimated annual consumption. '
                f'Based on {self.ANNUAL_HDD:.0f} annual HDD, {self.ANNUAL_CDD:.0f} annual CDD, '
                f'and ${self.COST_PER_KWH:.2f}/kWh.'
            ),
        }
    
    def generate_top_action_plans(
        self,
        limit: int = 10,
        exclude_special: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Generate action plans for the top-N buildings by estimated savings (ROI).
        """
        # Get all regular buildings
        candidates = [
            s for s in self.building_scores.values()
            if s.gross_area >= self.MIN_RANKING_AREA
            and (not exclude_special or s.building_type == 'regular')
        ]
        
        # Generate plans for all, then sort by total savings
        plans = []
        for s in candidates:
            plan = self.generate_action_plan(s.building_id)
            if plan and plan['total_estimated_kwh_savings'] > 0:
                plans.append(plan)
        
        plans.sort(key=lambda p: p['total_estimated_cost_savings'], reverse=True)
        return plans[:limit]
