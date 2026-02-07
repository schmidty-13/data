"""
API Schemas for Eco-Pulse
Pydantic models for request/response validation.

Updated with data quality flags (Task 6.1) and filtering support (Task 6.2).
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import date


# ============================================================
# Meter Schemas
# ============================================================

class MeterDetail(BaseModel):
    """Details for a single meter."""
    building_id: int = Field(..., description="Building identifier")
    utility: str = Field(..., description="Utility type (HEAT, ELECTRICITY, etc.)")
    degree_days_type: str = Field(..., description="HDD or CDD")
    slope: float = Field(..., description="Change-point slope")
    t_balance: float = Field(..., description="Change-point temperature (°F)")
    base_load: float = Field(..., description="Base energy load")
    total_energy: float = Field(..., description="Total energy consumption")
    r2: float = Field(..., description="Model R² score")
    confidence: str = Field(..., description="Model confidence (high/medium/low)")
    # Task 6.1: quality flags
    data_quality_flags: List[str] = Field(default_factory=list, description="Data quality flags")
    
    class Config:
        json_schema_extra = {
            "example": {
                "building_id": 123,
                "utility": "HEAT",
                "degree_days_type": "HDD",
                "slope": 3.45,
                "t_balance": 65.0,
                "base_load": 150.5,
                "total_energy": 125000.0,
                "r2": 0.85,
                "confidence": "high",
                "data_quality_flags": []
            }
        }


# ============================================================
# Building Schemas
# ============================================================

class BuildingSummary(BaseModel):
    """Summary information for a building (used in listings)."""
    building_id: int = Field(..., description="Building identifier")
    building_name: str = Field(..., description="Building name")
    gross_area: float = Field(..., description="Building gross area (sq ft)")
    building_type: str = Field('regular', description="Building type classification")
    
    heating_slope: Optional[float] = Field(None, description="Weighted heating slope")
    cooling_slope: Optional[float] = Field(None, description="Weighted cooling slope")
    
    total_energy: float = Field(..., description="Total energy consumption")
    eui: float = Field(..., description="Energy Use Intensity")
    
    heating_efficiency_score: Optional[float] = Field(None, description="Heating efficiency (0-100)")
    cooling_efficiency_score: Optional[float] = Field(None, description="Cooling efficiency (0-100)")
    overall_efficiency_score: Optional[float] = Field(None, description="Overall efficiency (0-100)")
    efficiency_rank: Optional[int] = Field(None, description="Rank among all buildings")
    
    avg_r2: float = Field(..., description="Average model R² score")
    avg_confidence: str = Field(..., description="Average model confidence")
    num_meters: int = Field(..., description="Number of meters analyzed")
    
    # Task 6.1: quality fields
    size_category: str = Field('medium', description="Size category (small/medium/large)")
    data_quality_flags: List[str] = Field(default_factory=list, description="Data quality flags")
    has_outliers: bool = Field(False, description="Whether building has outlier data")
    unit_converted: bool = Field(False, description="Whether units were converted (e.g. STEAM kg->kWh)")
    data_coverage: str = Field('full', description="Data coverage: full (heating+cooling), partial, or electricity_only")
    
    class Config:
        json_schema_extra = {
            "example": {
                "building_id": 123,
                "building_name": "Engineering Building",
                "gross_area": 75000.0,
                "building_type": "regular",
                "heating_slope": 3.45,
                "cooling_slope": 2.10,
                "total_energy": 250000.0,
                "eui": 3.33,
                "heating_efficiency_score": 72.5,
                "cooling_efficiency_score": 81.2,
                "overall_efficiency_score": 76.8,
                "efficiency_rank": 15,
                "avg_r2": 0.78,
                "avg_confidence": "high",
                "num_meters": 3,
                "size_category": "medium",
                "data_quality_flags": [],
                "has_outliers": False,
                "unit_converted": False,
                "data_coverage": "full"
            }
        }


class BuildingDetail(BuildingSummary):
    """Detailed information for a building (includes meters)."""
    composite_slope: Optional[float] = Field(None, description="Composite weighted slope")
    heating_energy: float = Field(..., description="Total heating-related energy")
    cooling_energy: float = Field(..., description="Total cooling-related energy")
    meters: List[MeterDetail] = Field(default_factory=list, description="Individual meter details")
    
    # Additional metadata
    building_age: Optional[int] = Field(None, description="Building age in years")
    latitude: Optional[float] = Field(None, description="Building latitude")
    longitude: Optional[float] = Field(None, description="Building longitude")
    
    class Config:
        json_schema_extra = {
            "example": {
                "building_id": 123,
                "building_name": "Engineering Building",
                "gross_area": 75000.0,
                "building_type": "regular",
                "heating_slope": 3.45,
                "cooling_slope": 2.10,
                "composite_slope": 2.85,
                "total_energy": 250000.0,
                "heating_energy": 150000.0,
                "cooling_energy": 100000.0,
                "eui": 3.33,
                "heating_efficiency_score": 72.5,
                "cooling_efficiency_score": 81.2,
                "overall_efficiency_score": 76.8,
                "efficiency_rank": 15,
                "avg_r2": 0.78,
                "avg_confidence": "high",
                "num_meters": 3,
                "building_age": 25,
                "latitude": 40.0,
                "longitude": -83.0,
                "meters": [],
                "size_category": "medium",
                "data_quality_flags": [],
                "has_outliers": False,
                "unit_converted": False,
                "data_coverage": "full"
            }
        }


# ============================================================
# V-Curve Schemas
# ============================================================

class VCurvePoint(BaseModel):
    temperature: float
    energy: float


class VCurveActual(BaseModel):
    temperatures: List[float] = Field(..., description="Temperature values (°F)")
    energy: List[float] = Field(..., description="Energy values")


class VCurveFitted(BaseModel):
    temperatures: List[float] = Field(..., description="Temperature values (°F)")
    energy: List[float] = Field(..., description="Fitted energy values")


class ChangePointInfo(BaseModel):
    temperature: float = Field(..., description="Change-point temperature (°F)")
    energy: float = Field(..., description="Base load at change-point")


class ModelInfo(BaseModel):
    model_type: str = Field(..., description="3P or 4P")
    t_balance: float = Field(..., description="Change-point temperature")
    base_load: float = Field(..., description="Base energy load")
    heating_slope: Optional[float] = Field(None, description="Heating slope")
    cooling_slope: Optional[float] = Field(None, description="Cooling slope")
    r2: float = Field(..., description="Model R² score")
    rmse: float = Field(..., description="Root Mean Squared Error")
    mae: float = Field(..., description="Mean Absolute Error")
    n_observations: int = Field(..., description="Number of observations")
    confidence: str = Field(..., description="Model confidence")
    validation_flags: List[str] = Field(default_factory=list, description="Model validation flags")


class VCurveData(BaseModel):
    building_id: int = Field(..., description="Building identifier")
    utility: str = Field(..., description="Utility type")
    actual: VCurveActual = Field(..., description="Actual data points")
    fitted: VCurveFitted = Field(..., description="Fitted line points")
    change_point: ChangePointInfo = Field(..., description="Change-point information")
    model_info: ModelInfo = Field(..., description="Model parameters")


# ============================================================
# Rankings & Summary Schemas
# ============================================================

class RankingsResponse(BaseModel):
    total: int = Field(..., description="Total number of buildings")
    buildings: List[BuildingSummary] = Field(..., description="Ranked building list")


class EfficiencyStats(BaseModel):
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    count: Optional[int] = None


class EUIStats(BaseModel):
    mean: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None


class ConfidenceDistribution(BaseModel):
    high: int = 0
    medium: int = 0
    low: int = 0


class QualityFlagCounts(BaseModel):
    """Counts of each quality flag across all buildings."""
    has_outliers: int = 0
    unit_converted: int = 0
    low_data_quality: int = 0
    null_score_imputed: int = 0
    extreme_slope: int = 0


class DataSummary(BaseModel):
    total_buildings: int = Field(..., description="Total number of buildings")
    total_meters: int = Field(..., description="Total number of meters")
    date_range: Dict[str, str] = Field(..., description="Date range of data")
    temperature_range: Dict[str, float] = Field(..., description="Temperature range")
    overall_efficiency: EfficiencyStats = Field(..., description="Overall efficiency stats")
    heating_efficiency: EfficiencyStats = Field(..., description="Heating efficiency stats")
    cooling_efficiency: EfficiencyStats = Field(..., description="Cooling efficiency stats")
    eui: EUIStats = Field(..., description="EUI statistics")
    confidence_distribution: ConfidenceDistribution = Field(..., description="Confidence distribution")
    quality_flag_counts: Dict[str, int] = Field(default_factory=dict, description="Quality flag counts")


# ============================================================
# Request Schemas
# ============================================================

class RankingsRequest(BaseModel):
    limit: Optional[int] = Field(None, ge=1, le=100, description="Maximum results")
    offset: Optional[int] = Field(0, ge=0, description="Offset for pagination")
    sort_by: Optional[str] = Field('overall_efficiency_score', description="Field to sort by")
    ascending: Optional[bool] = Field(False, description="Sort ascending")
    min_confidence: Optional[str] = Field(None, description="Minimum confidence level filter")


# ============================================================
# Action Plan Schemas
# ============================================================

class Recommendation(BaseModel):
    """A single retrofit recommendation for a building."""
    type: str = Field(..., description="Recommendation type (insulation_upgrade, hvac_upgrade, base_load_reduction, comprehensive_retrofit)")
    title: str = Field(..., description="Short title")
    description: str = Field(..., description="Detailed description with suggested actions")
    priority: str = Field(..., description="Priority level (critical, high, medium)")
    current_slope: Optional[float] = Field(None, description="Current slope value")
    target_slope: Optional[float] = Field(None, description="Target slope (median for size category)")
    current_base_load: Optional[float] = Field(None, description="Current base load (kWh/day)")
    target_base_load: Optional[float] = Field(None, description="Target base load (kWh/day)")
    estimated_kwh_savings: float = Field(0, description="Estimated annual kWh savings")
    estimated_cost_savings: float = Field(0, description="Estimated annual cost savings ($)")


class ActionPlan(BaseModel):
    """Complete action plan for a single building."""
    building_id: int = Field(..., description="Building identifier")
    building_name: str = Field(..., description="Building name")
    gross_area: float = Field(..., description="Building gross area (sq ft)")
    size_category: str = Field(..., description="Size category (small/medium/large)")
    current_efficiency_score: Optional[float] = Field(None, description="Current overall efficiency score")
    current_eui: float = Field(..., description="Current EUI (kWh/sqft)")
    heating_slope: Optional[float] = Field(None, description="Current heating slope (kWh/HDD)")
    cooling_slope: Optional[float] = Field(None, description="Current cooling slope (kWh/CDD)")
    recommendations: List[Recommendation] = Field(default_factory=list, description="List of recommendations")
    total_estimated_kwh_savings: float = Field(0, description="Total estimated annual kWh savings")
    total_estimated_cost_savings: float = Field(0, description="Total estimated annual cost savings ($)")
    methodology_note: str = Field("", description="Explanation of how savings were calculated")


class TopActionPlansResponse(BaseModel):
    """Response for top action plans endpoint."""
    total: int = Field(..., description="Total number of plans returned")
    plans: List[ActionPlan] = Field(..., description="Action plans sorted by estimated savings")


# ============================================================
# Solar Potential Schemas
# ============================================================

class SolarPotential(BaseModel):
    """Solar panel potential for a single building."""
    building_id: int = Field(..., description="Building identifier")
    building_name: str = Field(..., description="Building name")
    solar_potential_score: float = Field(..., description="Composite solar potential score (0-58)")
    solar_tier: str = Field(..., description="Solar tier (Top Priority, Excellent, Very Good, Good, Moderate, Fair, Low Priority)")

    # Physical characteristics
    solar_roof_area_m2: float = Field(..., description="Available roof area for panels (m2)")
    solar_max_panel_count: int = Field(..., description="Maximum solar panels that fit on the roof")
    solar_yearly_kwh_dc: float = Field(..., description="Estimated annual solar generation (kWh)")
    solar_sunshine_hours_year: float = Field(..., description="Annual sunshine hours at location")
    solar_shade_percent: float = Field(..., description="Percentage of roof shaded")

    # Savings
    estimated_annual_savings: float = Field(..., description="Estimated annual electricity cost savings ($)")
    solar_demand_offset_pct: float = Field(..., description="Percentage of current demand offset by solar")
    current_annual_kwh: Optional[float] = Field(None, description="Current annual energy consumption (kWh)")

    # Sub-scores (0-1 each)
    radiation_score: float = Field(0, description="Solar radiation sub-score")
    rooftop_score: float = Field(0, description="Rooftop suitability sub-score")
    shade_score: float = Field(0, description="Shade factor sub-score")
    demand_score: float = Field(0, description="Energy demand match sub-score")
    suitability_score: float = Field(0, description="Overall suitability sub-score")

    # Building context
    gross_area: Optional[float] = Field(None, description="Building gross area (sq ft)")
    floors_above_ground: Optional[int] = Field(None, description="Number of floors above ground")
    building_age: Optional[int] = Field(None, description="Building age in years")


class SolarTierBreakdown(BaseModel):
    """Count of buildings in each solar tier."""
    tier: str = Field(..., description="Tier name")
    count: int = Field(..., description="Number of buildings in this tier")
    total_yearly_kwh: float = Field(..., description="Total annual kWh for this tier")
    total_annual_savings: float = Field(..., description="Total annual savings for this tier ($)")


class SolarSummary(BaseModel):
    """Campus-wide solar potential summary."""
    total_buildings: int = Field(..., description="Number of buildings with solar data")
    total_yearly_kwh: float = Field(..., description="Total annual kWh generation potential")
    total_annual_savings: float = Field(..., description="Total annual electricity cost savings ($)")
    total_panel_count: int = Field(..., description="Total panels across all buildings")
    total_roof_area_m2: float = Field(..., description="Total available roof area (m2)")
    avg_demand_offset_pct: float = Field(..., description="Average demand offset percentage")
    tier_breakdown: List[SolarTierBreakdown] = Field(..., description="Breakdown by solar tier")


# ============================================================
# Error Schemas
# ============================================================

class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")
