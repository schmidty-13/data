"""
Eco-Pulse Data Validation Script (Task 7.1)

Runs a comprehensive battery of checks on the processed data to ensure
presentation-readiness.  Exit code 0 = all checks pass.

Usage:
    python validate_data.py [--data-dir PATH]
"""

import sys
import os
from pathlib import Path

# Ensure backend is on the path
sys.path.insert(0, str(Path(__file__).parent))

from models.data_processor import DataProcessor
from models.change_point import fit_all_meters
from models.aggregator import BuildingAggregator


# ====================================================================
#  Colour helpers (work in Windows Terminal / PowerShell)
# ====================================================================
def _pass(msg): return f"  [PASS] {msg}"
def _fail(msg): return f"  [FAIL] {msg}"
def _warn(msg): return f"  [WARN] {msg}"


def run_validation(data_dir: str):
    """
    Run the full pipeline then validate the output.
    Returns (passes, fails, warnings).
    """
    passes, fails, warnings_list = 0, 0, 0

    # ----------------------------------------------------------------
    # 1. Run pipeline
    # ----------------------------------------------------------------
    print("=" * 60)
    print("RUNNING FULL PIPELINE")
    print("=" * 60)

    processor = DataProcessor(data_dir)
    processor.load_energy_data()
    processor.load_building_metadata()
    processor.load_and_process_weather()
    merged_df = processor.join_all_data()

    model_results = fit_all_meters(merged_df)

    aggregator = BuildingAggregator(processor.building_df)
    building_scores = aggregator.aggregate_all_buildings(model_results, merged_df)

    scores = list(building_scores.values())
    total = len(scores)

    print("\n" + "=" * 60)
    print("VALIDATION CHECKS")
    print("=" * 60)

    # ----------------------------------------------------------------
    # CHECK 1: All buildings have efficiency scores (no NULLs)
    # ----------------------------------------------------------------
    print("\n--- Check 1: No NULL efficiency scores ---")
    null_scores = [s for s in scores if s.overall_efficiency_score is None]
    if len(null_scores) == 0:
        print(_pass(f"All {total} buildings have valid efficiency scores"))
        passes += 1
    else:
        print(_fail(f"{len(null_scores)}/{total} buildings still have NULL efficiency scores"))
        for s in null_scores[:5]:
            print(f"       Building {s.building_id}: {s.building_name}")
        fails += 1

    # ----------------------------------------------------------------
    # CHECK 2: Efficiency scores in valid range [0, 100]
    # ----------------------------------------------------------------
    print("\n--- Check 2: Scores in 0-100 range ---")
    out_of_range = [
        s for s in scores
        if s.overall_efficiency_score is not None
        and (s.overall_efficiency_score < 0 or s.overall_efficiency_score > 100)
    ]
    if len(out_of_range) == 0:
        print(_pass("All scores are in the 0-100 range"))
        passes += 1
    else:
        print(_fail(f"{len(out_of_range)} buildings have scores out of range"))
        fails += 1

    # ----------------------------------------------------------------
    # CHECK 3: Score distribution (not all 0 or all 100)
    # ----------------------------------------------------------------
    print("\n--- Check 3: Score distribution is meaningful ---")
    eff_scores = [s.overall_efficiency_score for s in scores if s.overall_efficiency_score is not None]
    zero_count = sum(1 for e in eff_scores if e == 0.0)
    hundred_count = sum(1 for e in eff_scores if e == 100.0)
    zero_pct = zero_count / len(eff_scores) * 100 if eff_scores else 0
    hundred_pct = hundred_count / len(eff_scores) * 100 if eff_scores else 0

    if zero_pct < 20 and hundred_pct < 20:
        print(_pass(f"Score distribution OK: {zero_count} at 0.0 ({zero_pct:.1f}%), {hundred_count} at 100.0 ({hundred_pct:.1f}%)"))
        passes += 1
    elif zero_pct >= 50:
        print(_fail(f"{zero_count}/{len(eff_scores)} ({zero_pct:.1f}%) buildings score 0.0 -- normalization issue"))
        fails += 1
    else:
        print(_warn(f"{zero_count} at 0.0 ({zero_pct:.1f}%), {hundred_count} at 100.0 ({hundred_pct:.1f}%)"))
        warnings_list += 1

    # ----------------------------------------------------------------
    # CHECK 4: No extreme energy values (> 1e12 kWh)
    # ----------------------------------------------------------------
    print("\n--- Check 4: No impossible energy values ---")
    extreme_energy = [s for s in scores if s.total_energy > 1e12]
    if len(extreme_energy) == 0:
        print(_pass("No buildings with energy > 1 trillion kWh"))
        passes += 1
    else:
        print(_fail(f"{len(extreme_energy)} buildings with extreme energy values"))
        for s in extreme_energy[:5]:
            print(f"       Building {s.building_id} ({s.building_name}): {s.total_energy:.2e} kWh")
        fails += 1

    # ----------------------------------------------------------------
    # CHECK 5: No extreme slopes (> 10,000)
    # ----------------------------------------------------------------
    print("\n--- Check 5: No impossible slopes ---")
    extreme_slopes = []
    for s in scores:
        max_sl = max(
            abs(s.heating_slope) if s.heating_slope else 0,
            abs(s.cooling_slope) if s.cooling_slope else 0
        )
        if max_sl > 10_000:
            extreme_slopes.append((s, max_sl))
    if len(extreme_slopes) == 0:
        print(_pass("No buildings with slopes > 10,000"))
        passes += 1
    else:
        print(_fail(f"{len(extreme_slopes)} buildings with extreme slopes"))
        for s, sl in extreme_slopes[:5]:
            print(f"       Building {s.building_id} ({s.building_name}): slope={sl:.1f}")
        fails += 1

    # ----------------------------------------------------------------
    # CHECK 6: Power plants excluded
    # ----------------------------------------------------------------
    print("\n--- Check 6: Power plants excluded ---")
    power_names = ['Power Plant', 'Substation', 'Power House']
    power_in_results = [
        s for s in scores
        if any(p.lower() in s.building_name.lower() for p in power_names)
    ]
    if len(power_in_results) == 0:
        print(_pass("No power plants/substations in results"))
        passes += 1
    else:
        print(_fail(f"{len(power_in_results)} power plants/substations still in results"))
        for s in power_in_results:
            print(f"       Building {s.building_id}: {s.building_name}")
        fails += 1

    # ----------------------------------------------------------------
    # CHECK 7: STEAM buildings have reasonable EUI
    # ----------------------------------------------------------------
    print("\n--- Check 7: STEAM buildings have reasonable EUI ---")
    steam_buildings = set()
    for s in scores:
        for m in s.meter_scores:
            if m.utility == 'STEAM':
                steam_buildings.add(s.building_id)
    steam_extreme = [s for s in scores if s.building_id in steam_buildings and s.eui > 500]
    if len(steam_extreme) == 0:
        print(_pass(f"All {len(steam_buildings)} STEAM buildings have EUI < 500"))
        passes += 1
    else:
        print(_warn(f"{len(steam_extreme)} STEAM buildings have EUI > 500"))
        for s in steam_extreme[:5]:
            print(f"       Building {s.building_id} ({s.building_name}): EUI={s.eui:.2f}")
        warnings_list += 1

    # ----------------------------------------------------------------
    # CHECK 8: Top 10 buildings have reasonable EUI (< 200)
    # ----------------------------------------------------------------
    print("\n--- Check 8: Top 10 buildings have reasonable EUI ---")
    ranked = sorted(scores, key=lambda x: x.efficiency_rank or 9999)
    top10 = ranked[:10]
    high_eui_top = [s for s in top10 if s.eui > 200]
    if len(high_eui_top) == 0:
        print(_pass("All top-10 buildings have EUI < 200"))
        passes += 1
    else:
        print(_warn(f"{len(high_eui_top)} top-10 buildings have EUI > 200"))
        for s in high_eui_top:
            print(f"       Rank {s.efficiency_rank}: {s.building_name} EUI={s.eui:.2f}")
        warnings_list += 1

    # ----------------------------------------------------------------
    # CHECK 9: Bottom 10 have high but not impossible slopes
    # ----------------------------------------------------------------
    print("\n--- Check 9: Bottom 10 buildings have high but reasonable slopes ---")
    bottom10 = ranked[-10:]
    impossible_bottom = []
    for s in bottom10:
        max_sl = max(
            abs(s.heating_slope) if s.heating_slope else 0,
            abs(s.cooling_slope) if s.cooling_slope else 0
        )
        if max_sl > 10_000:
            impossible_bottom.append((s, max_sl))
    if len(impossible_bottom) == 0:
        print(_pass("Bottom-10 buildings all have reasonable slopes"))
        passes += 1
    else:
        print(_fail(f"{len(impossible_bottom)} bottom-10 buildings have impossible slopes"))
        fails += 1

    # ----------------------------------------------------------------
    # CHECK 10: Data quality flags are present
    # ----------------------------------------------------------------
    print("\n--- Check 10: Quality flags are populated ---")
    flag_counts = {}
    for s in scores:
        for f in s.data_quality_flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    if flag_counts:
        print(_pass(f"Quality flags present: {flag_counts}"))
        passes += 1
    else:
        print(_warn("No quality flags found -- may indicate all data is clean"))
        warnings_list += 1

    # ----------------------------------------------------------------
    # CHECK 11: Unit conversion was applied
    # ----------------------------------------------------------------
    print("\n--- Check 11: Unit conversions applied ---")
    converted = sum(1 for s in scores if s.unit_converted)
    if converted > 0:
        print(_pass(f"{converted} buildings had units converted"))
        passes += 1
    else:
        print(_warn("No unit conversions recorded -- may be fine if no STEAM buildings"))
        warnings_list += 1

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"  Total checks: {passes + fails + warnings_list}")
    print(f"  Passed:   {passes}")
    print(f"  Failed:   {fails}")
    print(f"  Warnings: {warnings_list}")
    print(f"  Buildings: {total}")

    if eff_scores:
        import numpy as np
        print(f"\n  Efficiency score statistics:")
        print(f"    Mean:   {np.mean(eff_scores):.1f}")
        print(f"    Median: {np.median(eff_scores):.1f}")
        print(f"    Std:    {np.std(eff_scores):.1f}")
        print(f"    Min:    {min(eff_scores):.1f}")
        print(f"    Max:    {max(eff_scores):.1f}")

    if fails > 0:
        print(f"\n  STATUS: FAILED -- {fails} check(s) need attention")
        return 1
    elif warnings_list > 0:
        print(f"\n  STATUS: PASSED with {warnings_list} warning(s)")
        return 0
    else:
        print(f"\n  STATUS: ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Eco-Pulse Data Validation")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent.parent),
        help="Directory containing raw CSV data files"
    )
    args = parser.parse_args()
    sys.exit(run_validation(args.data_dir))
