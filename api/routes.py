"""
API Routes for Eco-Pulse
FastAPI endpoints for the energy audit dashboard.

Updated with quality flags in responses (Task 6.1) and filtering parameters (Task 6.2).
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
from .schemas import (
    BuildingSummary,
    BuildingDetail,
    MeterDetail,
    VCurveData,
    VCurveActual,
    VCurveFitted,
    ChangePointInfo,
    ModelInfo,
    RankingsResponse,
    DataSummary,
    EfficiencyStats,
    EUIStats,
    ConfidenceDistribution,
    ErrorResponse,
    ActionPlan,
    Recommendation,
    TopActionPlansResponse,
)

router = APIRouter(prefix="/api", tags=["buildings"])

# Global state
_data_processor = None
_model_results = None
_aggregator = None
_merged_df = None


def set_global_state(data_processor, model_results, aggregator, merged_df):
    global _data_processor, _model_results, _aggregator, _merged_df
    _data_processor = data_processor
    _model_results = model_results
    _aggregator = aggregator
    _merged_df = merged_df


def get_aggregator():
    if _aggregator is None:
        raise HTTPException(status_code=503, detail="Data not loaded. Please wait for initialization.")
    return _aggregator


# ============================================================
# Helper to convert BuildingScore → BuildingSummary
# ============================================================

def _to_summary(r) -> BuildingSummary:
    """Convert a BuildingScore dataclass to a BuildingSummary Pydantic model."""
    return BuildingSummary(
        building_id=r.building_id,
        building_name=r.building_name,
        gross_area=r.gross_area,
        building_type=r.building_type,
        heating_slope=r.heating_slope,
        cooling_slope=r.cooling_slope,
        total_energy=r.total_energy,
        eui=r.eui,
        heating_efficiency_score=r.heating_efficiency_score,
        cooling_efficiency_score=r.cooling_efficiency_score,
        overall_efficiency_score=r.overall_efficiency_score,
        efficiency_rank=r.efficiency_rank,
        avg_r2=r.avg_r2,
        avg_confidence=r.avg_confidence,
        num_meters=r.num_meters,
        size_category=r.size_category,
        data_quality_flags=r.data_quality_flags,
        has_outliers=r.has_outliers,
        unit_converted=r.unit_converted,
        data_coverage=r.data_coverage,
    )


# ============================================================
# Building Endpoints
# ============================================================

@router.get(
    "/buildings",
    response_model=RankingsResponse,
    summary="List all buildings",
    description="Get a list of all buildings with efficiency scores. Supports filtering."
)
async def list_buildings(
    limit: Optional[int] = Query(None, ge=1, le=500, description="Maximum results"),
    offset: Optional[int] = Query(0, ge=0, description="Offset for pagination"),
    sort_by: Optional[str] = Query('overall_efficiency_score', description="Field to sort by"),
    ascending: Optional[bool] = Query(False, description="Sort ascending"),
    min_confidence: Optional[str] = Query(None, description="Minimum confidence level (high, medium, low)"),
    # Task 6.2: new filter parameters
    exclude_special: Optional[bool] = Query(True, description="Exclude non-regular buildings (parking, greenhouse, stadium, etc.)"),
    exclude_outliers: Optional[bool] = Query(False, description="Exclude buildings with outlier data"),
    building_type: Optional[str] = Query(None, description="Filter by building type (regular, greenhouse, etc.)"),
    size_category: Optional[str] = Query(None, description="Filter by size (small, medium, large)"),
    aggregator=Depends(get_aggregator)
):
    rankings = aggregator.get_rankings(
        sort_by=sort_by,
        ascending=ascending,
        exclude_special=exclude_special,
        exclude_outliers=exclude_outliers,
        min_confidence=min_confidence,
    )
    
    # Additional filters
    if building_type:
        rankings = [r for r in rankings if r.building_type == building_type]
    if size_category:
        rankings = [r for r in rankings if r.size_category == size_category]
    
    total = len(rankings)
    if offset:
        rankings = rankings[offset:]
    if limit:
        rankings = rankings[:limit]
    
    return RankingsResponse(total=total, buildings=[_to_summary(r) for r in rankings])


@router.get(
    "/buildings/{building_id}",
    response_model=BuildingDetail,
    summary="Get building details",
    responses={404: {"model": ErrorResponse}}
)
async def get_building(building_id: int, aggregator=Depends(get_aggregator)):
    score = aggregator.get_building_score(building_id)
    if score is None:
        raise HTTPException(status_code=404, detail=f"Building {building_id} not found")
    
    metadata = aggregator.building_metadata.get(building_id, {})
    
    meters = [
        MeterDetail(
            building_id=m.building_id,
            utility=m.utility,
            degree_days_type=m.degree_days_type,
            slope=m.slope,
            t_balance=m.t_balance,
            base_load=m.base_load,
            total_energy=m.total_energy,
            r2=m.r2,
            confidence=m.confidence,
            data_quality_flags=m.data_quality_flags,
        )
        for m in score.meter_scores
    ]
    
    return BuildingDetail(
        building_id=score.building_id,
        building_name=score.building_name,
        gross_area=score.gross_area,
        building_type=score.building_type,
        heating_slope=score.heating_slope,
        cooling_slope=score.cooling_slope,
        composite_slope=score.composite_slope,
        total_energy=score.total_energy,
        heating_energy=score.heating_energy,
        cooling_energy=score.cooling_energy,
        eui=score.eui,
        heating_efficiency_score=score.heating_efficiency_score,
        cooling_efficiency_score=score.cooling_efficiency_score,
        overall_efficiency_score=score.overall_efficiency_score,
        efficiency_rank=score.efficiency_rank,
        avg_r2=score.avg_r2,
        avg_confidence=score.avg_confidence,
        num_meters=score.num_meters,
        meters=meters,
        building_age=metadata.get('building_age'),
        latitude=metadata.get('latitude'),
        longitude=metadata.get('longitude'),
        size_category=score.size_category,
        data_quality_flags=score.data_quality_flags,
        has_outliers=score.has_outliers,
        unit_converted=score.unit_converted,
        data_coverage=score.data_coverage,
    )


@router.get(
    "/buildings/{building_id}/vcurve",
    response_model=List[VCurveData],
    summary="Get V-curve data",
    responses={404: {"model": ErrorResponse}}
)
async def get_vcurve(
    building_id: int,
    utility: Optional[str] = Query(None, description="Filter by utility type"),
    aggregator=Depends(get_aggregator)
):
    global _model_results, _merged_df
    
    score = aggregator.get_building_score(building_id)
    if score is None:
        raise HTTPException(status_code=404, detail=f"Building {building_id} not found")
    
    vcurves = []
    for meter in score.meter_scores:
        if utility and meter.utility != utility:
            continue
        
        result_key = (building_id, meter.utility)
        if result_key not in _model_results:
            continue
        
        model_result = _model_results[result_key]
        
        meter_data = _merged_df[
            (_merged_df['building_id'] == building_id) &
            (_merged_df['utility'] == meter.utility)
        ]
        if meter_data.empty:
            continue
        
        from models.change_point import generate_vcurve_data
        vcurve_data = generate_vcurve_data(meter_data, model_result)
        
        vcurves.append(VCurveData(
            building_id=building_id,
            utility=meter.utility,
            actual=VCurveActual(
                temperatures=vcurve_data['actual']['temperatures'],
                energy=vcurve_data['actual']['energy']
            ),
            fitted=VCurveFitted(
                temperatures=vcurve_data['fitted']['temperatures'],
                energy=vcurve_data['fitted']['energy']
            ),
            change_point=ChangePointInfo(
                temperature=vcurve_data['change_point']['temperature'],
                energy=vcurve_data['change_point']['energy']
            ),
            model_info=ModelInfo(
                model_type=vcurve_data['model_info']['model_type'],
                t_balance=vcurve_data['model_info']['t_balance'],
                base_load=vcurve_data['model_info']['base_load'],
                heating_slope=vcurve_data['model_info']['heating_slope'],
                cooling_slope=vcurve_data['model_info']['cooling_slope'],
                r2=vcurve_data['model_info']['r2'],
                rmse=vcurve_data['model_info']['rmse'],
                mae=vcurve_data['model_info']['mae'],
                n_observations=vcurve_data['model_info']['n_observations'],
                confidence=vcurve_data['model_info']['confidence'],
                validation_flags=vcurve_data['model_info'].get('validation_flags', []),
            )
        ))
    
    if not vcurves:
        raise HTTPException(status_code=404, detail=f"No V-curve data found for building {building_id}")
    return vcurves


@router.get(
    "/buildings/{building_id}/meters",
    response_model=List[MeterDetail],
    summary="Get meter details",
    responses={404: {"model": ErrorResponse}}
)
async def get_meters(building_id: int, aggregator=Depends(get_aggregator)):
    score = aggregator.get_building_score(building_id)
    if score is None:
        raise HTTPException(status_code=404, detail=f"Building {building_id} not found")
    
    return [
        MeterDetail(
            building_id=m.building_id,
            utility=m.utility,
            degree_days_type=m.degree_days_type,
            slope=m.slope,
            t_balance=m.t_balance,
            base_load=m.base_load,
            total_energy=m.total_energy,
            r2=m.r2,
            confidence=m.confidence,
            data_quality_flags=m.data_quality_flags,
        )
        for m in score.meter_scores
    ]


# ============================================================
# Rankings Endpoint (Task 6.2: with filtering)
# ============================================================

@router.get(
    "/rankings",
    response_model=RankingsResponse,
    summary="Get efficiency rankings"
)
async def get_rankings(
    limit: Optional[int] = Query(20, ge=1, le=100, description="Maximum results"),
    sort_by: Optional[str] = Query('overall_efficiency_score', description="Field to sort by"),
    ascending: Optional[bool] = Query(False, description="Sort ascending"),
    efficiency_type: Optional[str] = Query(None, description="Filter: heating or cooling"),
    # Task 6.2: filters
    exclude_special: Optional[bool] = Query(True, description="Exclude power plants/substations (default True)"),
    exclude_outliers: Optional[bool] = Query(False, description="Exclude buildings with outlier data"),
    min_confidence: Optional[str] = Query(None, description="Minimum confidence (high, medium, low)"),
    building_type: Optional[str] = Query(None, description="Filter by building type"),
    size_category: Optional[str] = Query(None, description="Filter by size (small, medium, large)"),
    aggregator=Depends(get_aggregator)
):
    if efficiency_type == 'heating':
        sort_by = 'heating_efficiency_score'
    elif efficiency_type == 'cooling':
        sort_by = 'cooling_efficiency_score'
    
    rankings = aggregator.get_rankings(
        limit=None,  # we'll slice after filtering
        sort_by=sort_by,
        ascending=ascending,
        exclude_special=exclude_special,
        exclude_outliers=exclude_outliers,
        min_confidence=min_confidence,
    )
    
    # Additional type filters
    if efficiency_type == 'heating':
        rankings = [r for r in rankings if r.heating_efficiency_score is not None]
    elif efficiency_type == 'cooling':
        rankings = [r for r in rankings if r.cooling_efficiency_score is not None]
    if building_type:
        rankings = [r for r in rankings if r.building_type == building_type]
    if size_category:
        rankings = [r for r in rankings if r.size_category == size_category]
    
    total = len(rankings)
    if limit:
        rankings = rankings[:limit]
    
    return RankingsResponse(total=total, buildings=[_to_summary(r) for r in rankings])


# ============================================================
# Summary Endpoint
# ============================================================

@router.get("/summary", response_model=DataSummary, summary="Get data summary")
async def get_summary(aggregator=Depends(get_aggregator)):
    global _data_processor, _model_results
    
    data_summary = _data_processor.get_data_summary() if _data_processor else {}
    agg_summary = aggregator.get_summary_statistics()
    total_meters = len(_model_results) if _model_results else 0
    
    return DataSummary(
        total_buildings=agg_summary.get('total_buildings', 0),
        total_meters=total_meters,
        date_range=data_summary.get('date_range', {'start': '', 'end': ''}),
        temperature_range=data_summary.get('temperature_range', {'min': 0, 'max': 0}),
        overall_efficiency=EfficiencyStats(
            mean=agg_summary.get('overall_efficiency', {}).get('mean'),
            median=agg_summary.get('overall_efficiency', {}).get('median'),
            std=agg_summary.get('overall_efficiency', {}).get('std'),
            min=agg_summary.get('overall_efficiency', {}).get('min'),
            max=agg_summary.get('overall_efficiency', {}).get('max')
        ),
        heating_efficiency=EfficiencyStats(
            mean=agg_summary.get('heating_efficiency', {}).get('mean'),
            count=agg_summary.get('heating_efficiency', {}).get('count')
        ),
        cooling_efficiency=EfficiencyStats(
            mean=agg_summary.get('cooling_efficiency', {}).get('mean'),
            count=agg_summary.get('cooling_efficiency', {}).get('count')
        ),
        eui=EUIStats(
            mean=agg_summary.get('eui', {}).get('mean'),
            median=agg_summary.get('eui', {}).get('median'),
            min=agg_summary.get('eui', {}).get('min'),
            max=agg_summary.get('eui', {}).get('max')
        ),
        confidence_distribution=ConfidenceDistribution(
            high=agg_summary.get('confidence_distribution', {}).get('high', 0),
            medium=agg_summary.get('confidence_distribution', {}).get('medium', 0),
            low=agg_summary.get('confidence_distribution', {}).get('low', 0)
        ),
        quality_flag_counts=agg_summary.get('quality_flag_counts', {}),
    )


# ============================================================
# Action Plan Endpoints
# ============================================================

@router.get(
    "/buildings/{building_id}/action-plan",
    response_model=ActionPlan,
    summary="Get action plan for a building",
    description="Generate physics-based retrofit recommendations with estimated savings.",
    responses={404: {"model": ErrorResponse}}
)
async def get_building_action_plan(
    building_id: int,
    aggregator=Depends(get_aggregator)
):
    plan = aggregator.generate_action_plan(building_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"Building {building_id} not found")
    
    return ActionPlan(
        building_id=plan['building_id'],
        building_name=plan['building_name'],
        gross_area=plan['gross_area'],
        size_category=plan['size_category'],
        current_efficiency_score=plan['current_efficiency_score'],
        current_eui=plan['current_eui'],
        heating_slope=plan['heating_slope'],
        cooling_slope=plan['cooling_slope'],
        recommendations=[Recommendation(**r) for r in plan['recommendations']],
        total_estimated_kwh_savings=plan['total_estimated_kwh_savings'],
        total_estimated_cost_savings=plan['total_estimated_cost_savings'],
        methodology_note=plan['methodology_note'],
    )


@router.get(
    "/action-plans/top",
    response_model=TopActionPlansResponse,
    summary="Get top ROI action plans",
    description="Get action plans for buildings with the highest estimated savings."
)
async def get_top_action_plans(
    limit: Optional[int] = Query(10, ge=1, le=50, description="Number of plans to return"),
    exclude_special: Optional[bool] = Query(True, description="Exclude non-regular buildings"),
    aggregator=Depends(get_aggregator)
):
    plans = aggregator.generate_top_action_plans(limit=limit, exclude_special=exclude_special)
    
    action_plans = [
        ActionPlan(
            building_id=p['building_id'],
            building_name=p['building_name'],
            gross_area=p['gross_area'],
            size_category=p['size_category'],
            current_efficiency_score=p['current_efficiency_score'],
            current_eui=p['current_eui'],
            heating_slope=p['heating_slope'],
            cooling_slope=p['cooling_slope'],
            recommendations=[Recommendation(**r) for r in p['recommendations']],
            total_estimated_kwh_savings=p['total_estimated_kwh_savings'],
            total_estimated_cost_savings=p['total_estimated_cost_savings'],
            methodology_note=p['methodology_note'],
        )
        for p in plans
    ]
    
    return TopActionPlansResponse(total=len(action_plans), plans=action_plans)


# ============================================================
# Health Check
# ============================================================

@router.get("/health", summary="Health check")
async def health_check():
    return {
        "status": "healthy",
        "data_loaded": _aggregator is not None,
        "buildings_count": len(_aggregator.building_scores) if _aggregator else 0,
        "meters_count": len(_model_results) if _model_results else 0
    }
