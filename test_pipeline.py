"""
Test script for the complete Eco-Pulse pipeline.
"""

import sys
sys.path.insert(0, '.')

from models.data_processor import DataProcessor
from models.change_point import fit_all_meters
from models.aggregator import BuildingAggregator

def main():
    # Load data
    print("Loading data...")
    processor = DataProcessor('..')
    processor.load_energy_data()
    processor.load_building_metadata()
    processor.load_and_process_weather()
    merged = processor.join_all_data()

    # Fit models
    print("\nFitting change-point models...")
    model_results = fit_all_meters(merged)

    # Aggregate building scores
    print("\nAggregating building scores...")
    aggregator = BuildingAggregator(processor.building_df)
    building_scores = aggregator.aggregate_all_buildings(model_results, merged)

    # Show results
    print("=" * 60)
    print("PIPELINE TEST COMPLETE")
    print("=" * 60)

    # Summary
    summary = aggregator.get_summary_statistics()
    print(f"\nTotal buildings: {summary['total_buildings']}")
    print(f"Overall efficiency mean: {summary['overall_efficiency']['mean']:.1f}")
    print(f"Confidence distribution: {summary['confidence_distribution']}")

    # Top 5 buildings
    rankings = aggregator.get_rankings(limit=5)
    print("\nTop 5 Most Efficient Buildings:")
    for r in rankings:
        name = r.building_name[:40] if r.building_name else f"Building {r.building_id}"
        score = r.overall_efficiency_score or 0
        print(f"  {r.efficiency_rank}. {name} - Score: {score:.1f}")

    # Bottom 5 buildings
    rankings_worst = aggregator.get_rankings(limit=5, ascending=True)
    print("\nBottom 5 Least Efficient Buildings:")
    for r in rankings_worst:
        name = r.building_name[:40] if r.building_name else f"Building {r.building_id}"
        score = r.overall_efficiency_score or 0
        print(f"  {r.efficiency_rank}. {name} - Score: {score:.1f}")

    # Save processed results
    print("\n" + "=" * 60)
    print("Saving processed data...")
    processor.save_processed_data()
    
    # Save building scores to CSV
    scores_df = aggregator.to_dataframe()
    scores_df.to_csv('../backend/data/processed/building_scores.csv', index=False)
    print("Saved building scores to building_scores.csv")
    
    print("\nAll tests passed!")
    return processor, model_results, aggregator

if __name__ == "__main__":
    main()
