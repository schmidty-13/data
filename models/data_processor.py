"""
Data Processor Module for Eco-Pulse
Handles loading, cleaning, and joining of energy, building, and weather data.

Production-ready with:
- Unit conversion (STEAM kg → kWh, kW → kWh)
- Statistical outlier detection (IQR-based)
- Special building exclusion (power plants, substations)
- Building type classification
- Data quality flags
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DataProcessor:
    """
    Handles all data loading, cleaning, and joining operations for Eco-Pulse.
    """
    
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
    
    # Utility types to exclude from analysis (non-standard or overlapping)
    EXCLUDED_UTILITIES = {'STEAMRATE', 'OIL28SEC', 'COOLING_POWER'}
    
    # Heating utilities that can overlap (keep only the largest per building)
    HDD_UTILITIES = {'HEAT', 'STEAM', 'GAS'}
    
    # ----------------------------------------------------------------
    # Unit conversion factors (Task 1.1 & 1.2)
    # ----------------------------------------------------------------
    # 1 kg of steam ≈ 0.7 kWh thermal energy (at ~150 psi typical campus steam)
    STEAM_KG_TO_KWH = 0.7
    # kW readings are 15-minute interval power; each row = 15 min = 0.25 h
    KW_TO_KWH_FACTOR = 0.25
    # kg/hour steam rate – convert mass flow to energy per hour
    KG_PER_HOUR_TO_KWH = 0.7  # same thermal factor, already per-hour
    
    # ----------------------------------------------------------------
    # Special-building exclusion patterns (Task 3.1 & 3.2)
    # ----------------------------------------------------------------
    EXCLUDE_NAME_PATTERNS = [
        'Power Plant', 'Substation', 'Power House',
        'Central Service Building', 'Electric Substation',
        'Chilled Water Plant', 'Chiller Plant',
    ]
    
    # Explicit IDs to exclude (identified from data inspection)
    EXCLUDE_BUILDING_IDS = {
        69,   # McCracken Power Plant
        77,   # Central Service Building
        79,   # OSU Electric Substation
        127,  # Smith Electrical Substation
        130,  # Power House
        134,  # Substation West Campus
    }
    
    # Building-type classification keywords (Task 3.2)
    BUILDING_TYPE_KEYWORDS = {
        'power_generation': ['Power Plant', 'Power House'],
        'infrastructure':   ['Substation', 'Central Service'],
        'greenhouse':       ['Greenhouse', 'Hoop Greenhouse'],
        'parking':          ['Parking Garage'],
        'stadium':          ['Stadium', 'Arena'],
    }
    
    # ----------------------------------------------------------------
    # Outlier thresholds (Task 2.1 & 2.2)
    # ----------------------------------------------------------------
    IQR_MULTIPLIER = 3.0          # Flag data > Q3 + 3*IQR
    MAX_DAILY_KWH = 5_000_000     # 5 M kWh/day hard cap (pre-dedup, single meter reading)
    MAX_DAILY_BUILDING_KWH = 200_000  # 200K kWh/day hard cap per building-utility after aggregation
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.energy_df: Optional[pd.DataFrame] = None
        self.building_df: Optional[pd.DataFrame] = None
        self.weather_df: Optional[pd.DataFrame] = None
        self.merged_df: Optional[pd.DataFrame] = None
        
    # ================================================================
    #  Phase 1: Load & clean energy data  (Tasks 1.1, 1.2, 2.1)
    # ================================================================
    def load_energy_data(self) -> pd.DataFrame:
        print("Loading energy data from monthly CSV files...")
        
        meter_files = list(self.data_dir.glob("meter-readings-*.csv"))
        print(f"  Found {len(meter_files)} monthly files")
        if not meter_files:
            raise FileNotFoundError(f"No meter-readings-*.csv files found in {self.data_dir}")
        
        dfs = []
        for file in sorted(meter_files):
            df = pd.read_csv(file)
            dfs.append(df)
            print(f"  Loaded {file.name}: {len(df):,} rows")
        
        energy_df = pd.concat(dfs, ignore_index=True)
        initial_count = len(energy_df)
        print(f"  Total rows before cleaning: {initial_count:,}")
        
        # ---- basic filtering ----
        energy_df = energy_df[
            (energy_df['readingwindowsum'].notna()) & 
            (energy_df['readingwindowsum'] > 0)
        ]
        print(f"  Filtered out {initial_count - len(energy_df):,} zero/null readings")
        
        excluded_mask = energy_df['utility'].isin(self.EXCLUDED_UTILITIES)
        energy_df = energy_df[~excluded_mask]
        print(f"  Excluded {excluded_mask.sum():,} rows with non-standard utilities")
        
        # ---- Task 1.1 & 1.2: Unit Conversion ----
        energy_df = self._convert_units(energy_df)
        
        # ---- building id ----
        energy_df['simscode'] = pd.to_numeric(energy_df['simscode'], errors='coerce')
        energy_df['building_id'] = energy_df['simscode'].astype('Int64')
        null_id_count = energy_df['building_id'].isna().sum()
        energy_df = energy_df.dropna(subset=['building_id'])
        print(f"  Dropped {null_id_count:,} rows with missing building IDs")
        
        # ---- date ----
        energy_df['date'] = pd.to_datetime(energy_df['readingwindowstart']).dt.date
        
        # ---- Task 2.1: Statistical outlier detection (global) ----
        pre_outlier = len(energy_df)
        energy_df = self._remove_outliers_global(energy_df, col='readingwindowsum')
        print(f"  Removed {pre_outlier - len(energy_df):,} global outlier rows")
        
        # ---- select columns (keep meterid for deduplication) ----
        energy_df = energy_df[[
            'building_id', 'meterid', 'sitename', 'utility', 'date',
            'readingwindowsum', 'readingwindowmean', 'readingunits'
        ]].copy()
        energy_df.columns = [
            'building_id', 'meter_id', 'site_name', 'utility', 'date',
            'energy_sum', 'energy_mean', 'units'
        ]
        
        # ---- Deduplicate hourly rows to daily per-meter values ----
        # Raw data has 24 hourly rows per (meter, day), each carrying the
        # same readingwindowsum (daily window total).  Keep one row per
        # (building, utility, meter, date).
        pre_dedup = len(energy_df)
        energy_df = energy_df.drop_duplicates(
            subset=['building_id', 'utility', 'meter_id', 'date'], keep='first'
        )
        print(f"  Deduplicated hourly rows: {pre_dedup:,} -> {len(energy_df):,} "
              f"(removed {pre_dedup - len(energy_df):,} duplicate daily entries)")
        
        # ---- Aggregate across meters to building-utility-daily totals ----
        # Buildings may have multiple physical meters for the same utility.
        # Sum their daily readings to get a single daily total per
        # (building, utility, date).
        pre_agg = len(energy_df)
        num_meters_per_group = energy_df.groupby(
            ['building_id', 'utility', 'date']
        )['meter_id'].transform('nunique')
        multi_meter_rows = (num_meters_per_group > 1).sum()
        
        energy_df = energy_df.groupby(
            ['building_id', 'site_name', 'utility', 'date', 'units'],
            as_index=False
        ).agg({
            'energy_sum': 'sum',     # sum across meters for same utility
            'energy_mean': 'sum',    # sum of means across meters
        })
        print(f"  Aggregated multi-meter readings: {pre_agg:,} -> {len(energy_df):,} "
              f"({multi_meter_rows:,} rows had multiple meters)")
        
        # ---- Task 2.2: Per-building-meter outlier capping ----
        energy_df = self._cap_per_building_outliers(energy_df)
        
        # ---- Hard daily cap per building-utility ----
        # Catches corrupt readings (e.g. cumulative meter totals reported
        # as daily sums) that slip past percentile-based capping.
        hard_cap_mask = energy_df['energy_sum'] > self.MAX_DAILY_BUILDING_KWH
        hard_cap_count = int(hard_cap_mask.sum())
        if hard_cap_count > 0:
            energy_df.loc[hard_cap_mask, 'energy_sum'] = self.MAX_DAILY_BUILDING_KWH
            print(f"  Hard-capped {hard_cap_count:,} daily readings at "
                  f"{self.MAX_DAILY_BUILDING_KWH:,} kWh")
        
        # ---- Deduplicate overlapping heating utilities ----
        # HEAT, STEAM, and GAS all measure heating at different system
        # points (thermal output, campus steam, local fuel).  For each
        # building keep only the one with the highest total energy to
        # prevent double-counting.
        energy_df = self._dedup_heating_utilities(energy_df)
        
        print(f"  Final energy rows: {len(energy_df):,}")
        print(f"  Unique buildings: {energy_df['building_id'].nunique()}")
        print(f"  Unique utilities: {energy_df['utility'].unique().tolist()}")
        print(f"  Date range: {energy_df['date'].min()} to {energy_df['date'].max()}")
        
        self.energy_df = energy_df
        return energy_df
    
    # ----------------------------------------------------------------
    def _convert_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """Task 1.1 & 1.2: Convert all readings to kWh."""
        original_units = df['readingunits'].value_counts()
        print("  Unit conversion:")
        
        # STEAM / kg → kWh
        mask_kg = df['readingunits'] == 'kg'
        count_kg = mask_kg.sum()
        if count_kg > 0:
            df.loc[mask_kg, 'readingwindowsum'] *= self.STEAM_KG_TO_KWH
            df.loc[mask_kg, 'readingwindowmean'] *= self.STEAM_KG_TO_KWH
            df.loc[mask_kg, 'readingunits'] = 'kWh'
            print(f"    Converted {count_kg:,} kg rows -> kWh  (x{self.STEAM_KG_TO_KWH})")
        
        # kW (power) → kWh  (assuming 15-min intervals → ×0.25)
        mask_kw = df['readingunits'] == 'kW'
        count_kw = mask_kw.sum()
        if count_kw > 0:
            df.loc[mask_kw, 'readingwindowsum'] *= self.KW_TO_KWH_FACTOR
            df.loc[mask_kw, 'readingwindowmean'] *= self.KW_TO_KWH_FACTOR
            df.loc[mask_kw, 'readingunits'] = 'kWh'
            print(f"    Converted {count_kw:,} kW rows -> kWh  (x{self.KW_TO_KWH_FACTOR})")
        
        # kg/hour → kWh
        mask_kgh = df['readingunits'] == 'kg/hour'
        count_kgh = mask_kgh.sum()
        if count_kgh > 0:
            df.loc[mask_kgh, 'readingwindowsum'] *= self.KG_PER_HOUR_TO_KWH
            df.loc[mask_kgh, 'readingwindowmean'] *= self.KG_PER_HOUR_TO_KWH
            df.loc[mask_kgh, 'readingunits'] = 'kWh'
            print(f"    Converted {count_kgh:,} kg/hour rows -> kWh  (x{self.KG_PER_HOUR_TO_KWH})")
        
        remaining = df['readingunits'].value_counts()
        print(f"    Final unit distribution: {remaining.to_dict()}")
        return df
    
    # ----------------------------------------------------------------
    def _remove_outliers_global(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Task 2.1: IQR-based global outlier removal + hard cap."""
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_fence = q3 + self.IQR_MULTIPLIER * iqr
        
        # Also apply a hard-cap (no single daily meter reading > 5M kWh)
        threshold = min(upper_fence, self.MAX_DAILY_KWH)
        
        outlier_mask = df[col] > threshold
        print(f"  Outlier detection: IQR fence={upper_fence:,.1f}, hard cap={self.MAX_DAILY_KWH:,}")
        print(f"  Flagged {outlier_mask.sum():,} outlier rows (>{threshold:,.1f})")
        
        return df[~outlier_mask].copy()
    
    # ----------------------------------------------------------------
    def _cap_per_building_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Task 2.2: Per-building-meter capping at 99th percentile (vectorized)."""
        # Compute the 99th-percentile threshold per (building, utility) group
        p99 = df.groupby(['building_id', 'utility'])['energy_sum'].transform('quantile', 0.99)
        
        over_mask = df['energy_sum'] > p99
        capped_count = int(over_mask.sum())
        
        df.loc[over_mask, 'energy_sum'] = p99[over_mask]
        
        print(f"  Capped {capped_count:,} per-building outlier values at 99th percentile")
        return df
    
    # ----------------------------------------------------------------
    def _dedup_heating_utilities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only ONE heating utility per building.
        
        HEAT, STEAM, and GAS all measure heating energy at different
        points in the system (thermal output, campus steam supply,
        local fuel).  For each building that has more than one of these,
        keep the utility with the highest total energy and drop the rest.
        """
        hdd_mask = df['utility'].isin(self.HDD_UTILITIES)
        hdd_data = df[hdd_mask]
        non_hdd_data = df[~hdd_mask]
        
        if hdd_data.empty:
            return df
        
        # Total energy per (building, utility) for HDD utilities only
        hdd_totals = (
            hdd_data
            .groupby(['building_id', 'utility'])['energy_sum']
            .sum()
            .reset_index()
        )
        
        # For each building, find the HDD utility with the most energy
        idx_max = hdd_totals.groupby('building_id')['energy_sum'].idxmax()
        best_hdd = hdd_totals.loc[idx_max, ['building_id', 'utility']]
        best_hdd_set = set(zip(best_hdd['building_id'], best_hdd['utility']))
        
        # Count how many buildings had duplicates dropped
        buildings_with_multi = (
            hdd_totals.groupby('building_id')['utility'].nunique()
        )
        multi_count = (buildings_with_multi > 1).sum()
        
        # Filter: keep only the primary HDD utility per building
        keep_mask = hdd_data.apply(
            lambda r: (r['building_id'], r['utility']) in best_hdd_set, axis=1
        )
        kept_hdd = hdd_data[keep_mask]
        dropped = len(hdd_data) - len(kept_hdd)
        
        result = pd.concat([non_hdd_data, kept_hdd], ignore_index=True)
        print(f"  Heating dedup: kept 1 of HEAT/STEAM/GAS per building "
              f"({multi_count} buildings had overlaps, dropped {dropped:,} rows)")
        return result
    
    # ================================================================
    #  Phase 3: Load building metadata  (Tasks 3.1, 3.2)
    # ================================================================
    def load_building_metadata(self) -> pd.DataFrame:
        print("\nLoading building metadata...")
        
        metadata_path = self.data_dir / "building_metadata.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Building metadata file not found: {metadata_path}")
        
        building_df = pd.read_csv(metadata_path)
        print(f"  Loaded {len(building_df):,} buildings")
        
        building_df['buildingnumber'] = building_df['buildingnumber'].astype(int)
        
        # grossarea
        missing_area = building_df['grossarea'].isna().sum()
        zero_area = (building_df['grossarea'] == 0).sum()
        if missing_area > 0 or zero_area > 0:
            print(f"  Warning: {missing_area} missing + {zero_area} zero grossarea values")
            median_area = building_df.loc[building_df['grossarea'] > 0, 'grossarea'].median()
            building_df['grossarea_imputed'] = (building_df['grossarea'].isna()) | (building_df['grossarea'] == 0)
            building_df['grossarea'] = building_df['grossarea'].replace(0, np.nan).fillna(median_area)
        else:
            building_df['grossarea_imputed'] = False
        
        # building age
        building_df['constructiondate'] = pd.to_datetime(building_df['constructiondate'], errors='coerce')
        building_df['building_age'] = datetime.now().year - building_df['constructiondate'].dt.year
        
        # ---- Task 3.2: Building type classification ----
        building_df['building_type'] = 'regular'
        for btype, keywords in self.BUILDING_TYPE_KEYWORDS.items():
            for kw in keywords:
                mask = building_df['buildingname'].str.contains(kw, case=False, na=False)
                building_df.loc[mask, 'building_type'] = btype
        
        # Force-tag explicitly excluded IDs as infrastructure/power_generation
        building_df.loc[
            building_df['buildingnumber'].isin(self.EXCLUDE_BUILDING_IDS),
            'building_type'
        ] = 'infrastructure'
        
        type_counts = building_df['building_type'].value_counts().to_dict()
        print(f"  Building type distribution: {type_counts}")
        
        # ---- Task 3.1: Exclude special buildings ----
        pre_exclude = len(building_df)
        # Exclude by name pattern
        for pattern in self.EXCLUDE_NAME_PATTERNS:
            building_df = building_df[
                ~building_df['buildingname'].str.contains(pattern, case=False, na=False)
            ]
        # Exclude by explicit ID
        building_df = building_df[~building_df['buildingnumber'].isin(self.EXCLUDE_BUILDING_IDS)]
        excluded_count = pre_exclude - len(building_df)
        print(f"  Excluded {excluded_count} special buildings (power plants, substations)")
        
        # select columns
        building_df = building_df[[
            'buildingnumber', 'buildingname', 'grossarea', 'grossarea_imputed',
            'constructiondate', 'building_age', 'latitude', 'longitude', 'building_type'
        ]].copy()
        building_df.columns = [
            'building_id', 'building_name', 'gross_area', 'gross_area_imputed',
            'construction_date', 'building_age', 'latitude', 'longitude', 'building_type'
        ]
        
        print(f"  Final buildings: {len(building_df):,}")
        print(f"  Area range: {building_df['gross_area'].min():,.0f} - {building_df['gross_area'].max():,.0f} sq ft")
        
        self.building_df = building_df
        return building_df
    
    # ================================================================
    #  Weather data  (unchanged)
    # ================================================================
    def load_and_process_weather(self) -> pd.DataFrame:
        print("\nLoading and processing weather data...")
        
        weather_path = self.data_dir / "weather_data_hourly_2025.csv"
        if not weather_path.exists():
            raise FileNotFoundError(f"Weather data file not found: {weather_path}")
        
        weather_df = pd.read_csv(weather_path)
        print(f"  Loaded {len(weather_df):,} hourly records")
        
        weather_df['datetime'] = pd.to_datetime(weather_df['date'])
        weather_df['date'] = weather_df['datetime'].dt.date
        
        daily_weather = weather_df.groupby('date').agg({
            'temperature_2m': 'mean',
            'relative_humidity_2m': 'mean',
            'wind_speed_10m': 'mean',
            'cloud_cover': 'mean'
        }).reset_index()
        daily_weather.columns = ['date', 'avg_temp', 'avg_humidity', 'avg_wind_speed', 'avg_cloud_cover']
        
        BALANCE_TEMP = 65.0
        daily_weather['HDD'] = np.maximum(0, BALANCE_TEMP - daily_weather['avg_temp'])
        daily_weather['CDD'] = np.maximum(0, daily_weather['avg_temp'] - BALANCE_TEMP)
        
        print(f"  Aggregated to {len(daily_weather):,} daily records")
        print(f"  Temp range: {daily_weather['avg_temp'].min():.1f}F - {daily_weather['avg_temp'].max():.1f}F")
        print(f"  Total HDD: {daily_weather['HDD'].sum():,.1f}  CDD: {daily_weather['CDD'].sum():,.1f}")
        
        self.weather_df = daily_weather
        return daily_weather
    
    # ================================================================
    #  Join all data
    # ================================================================
    def join_all_data(self) -> pd.DataFrame:
        print("\nJoining all data sources...")
        
        if self.energy_df is None:
            self.load_energy_data()
        if self.building_df is None:
            self.load_building_metadata()
        if self.weather_df is None:
            self.load_and_process_weather()
        
        merged = self.energy_df.merge(self.building_df, on='building_id', how='inner')
        print(f"  After building join: {len(merged):,} rows, {merged['building_id'].nunique()} buildings")
        
        merged['date'] = pd.to_datetime(merged['date']).dt.date
        self.weather_df['date'] = pd.to_datetime(self.weather_df['date']).dt.date
        
        merged = merged.merge(self.weather_df, on='date', how='inner')
        print(f"  After weather join: {len(merged):,} rows")
        
        # EUI
        merged['EUI'] = merged['energy_sum'] / merged['gross_area']
        
        # degree-days type
        merged['degree_days_type'] = merged['utility'].map(
            lambda x: self.UTILITY_MODELS.get(x, {}).get('degree_days', 'UNKNOWN')
        )
        unknown_count = (merged['degree_days_type'] == 'UNKNOWN').sum()
        if unknown_count > 0:
            print(f"  Warning: {unknown_count:,} rows with unknown utility types removed")
            merged = merged[merged['degree_days_type'] != 'UNKNOWN']
        
        merged['degree_days'] = np.where(
            merged['degree_days_type'] == 'HDD', merged['HDD'], merged['CDD']
        )
        
        print(f"  Final merged rows: {len(merged):,}")
        print(f"  Unique buildings: {merged['building_id'].nunique()}")
        print(f"  Date range: {merged['date'].min()} to {merged['date'].max()}")
        
        # Summary
        print("\n  Summary by utility type:")
        utility_summary = merged.groupby('utility').agg(
            buildings=('building_id', 'nunique'),
            total_energy=('energy_sum', 'sum'),
            mean_energy=('energy_sum', 'mean'),
            mean_eui=('EUI', 'mean')
        ).round(2)
        print(utility_summary.to_string())
        
        self.merged_df = merged
        return merged
    
    # ================================================================
    #  Utility methods  (unchanged API)
    # ================================================================
    def get_building_meter_data(self, building_id: int, utility: Optional[str] = None) -> pd.DataFrame:
        if self.merged_df is None:
            raise ValueError("Data not loaded. Call join_all_data() first.")
        mask = self.merged_df['building_id'] == building_id
        if utility:
            mask &= self.merged_df['utility'] == utility
        return self.merged_df[mask].copy()
    
    def get_all_building_meters(self) -> Dict[int, List[str]]:
        if self.merged_df is None:
            raise ValueError("Data not loaded. Call join_all_data() first.")
        return self.merged_df.groupby('building_id')['utility'].unique().apply(list).to_dict()
    
    def save_processed_data(self, output_path: Optional[str] = None) -> str:
        if self.merged_df is None:
            raise ValueError("Data not loaded. Call join_all_data() first.")
        if output_path is None:
            output_path = self.data_dir / "backend" / "data" / "processed" / "merged_data.csv"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.merged_df.to_csv(output_path, index=False)
        print(f"\nSaved processed data to: {output_path}")
        return str(output_path)
    
    def get_data_summary(self) -> Dict:
        if self.merged_df is None:
            raise ValueError("Data not loaded. Call join_all_data() first.")
        return {
            'total_rows': len(self.merged_df),
            'unique_buildings': self.merged_df['building_id'].nunique(),
            'unique_utilities': self.merged_df['utility'].nunique(),
            'utility_types': self.merged_df['utility'].unique().tolist(),
            'date_range': {
                'start': str(self.merged_df['date'].min()),
                'end': str(self.merged_df['date'].max())
            },
            'temperature_range': {
                'min': float(self.merged_df['avg_temp'].min()),
                'max': float(self.merged_df['avg_temp'].max())
            },
            'total_hdd': float(self.merged_df.groupby('date')['HDD'].first().sum()),
            'total_cdd': float(self.merged_df.groupby('date')['CDD'].first().sum()),
        }


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent.parent.parent)
    processor = DataProcessor(data_dir)
    processor.load_energy_data()
    processor.load_building_metadata()
    processor.load_and_process_weather()
    processor.join_all_data()
    print("\n" + "=" * 60)
    print("DATA PROCESSING COMPLETE")
    print("=" * 60)
    summary = processor.get_data_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
