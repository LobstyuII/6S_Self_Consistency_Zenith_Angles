# ==================== external_validator.py ====================
"""
简化版外部验证器 - 支持所有6个波段
"""

# ==================== external_validator.py ====================
"""
Simplified External Validator - Supports all 6 bands
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


# ==================== Configuration ====================
class Config:
    # Data paths
    DATA_PATHS = {
        "hourly_sozSR": "D:/H8_data/Hourly_sozSR_Angles/",
        "merra2_slv": "D:/H8_data/MERRA2_slv/",
        "merra2_aer": "D:/H8_data/MERRA2_aer/",
        "lucc": "D:/H8_data/LC_2015_2024.nc",
        "luts": "D:/H8_data/LUTs.nc"
    }

    # Band configuration (1-6 bands)
    BAND_CONFIG = {
        '01': {'wavelength': 0.47, 'name': 'band1'},
        '02': {'wavelength': 0.51, 'name': 'band2'},
        '03': {'wavelength': 0.64, 'name': 'band3'},
        '04': {'wavelength': 0.86, 'name': 'band4'},
        '05': {'wavelength': 1.60, 'name': 'band5'},
        '06': {'wavelength': 2.30, 'name': 'band6'}
    }

    # Model expected feature order (based on your training)
    EXPECTED_FEATURES = [
        'sza', 'vza', 'raa', 'aod550', 'h2o', 'o3', 'wavelength',
        'rho_toa', 'rho_retrieved', 'cos_sza', 'sin_sza', 'cos_vza',
        'sin_vza', 'cos_raa', 'sin_raa', 'scattering_angle',
        'airmass_sza', 'airmass_vza', 'total_airmass', 'vza_sza_ratio',
        'vza_minus_sza', 'vza_plus_sza', 'aod_airmass', 'aod_wavelength',
        'rho_ratio', 'rho_diff', 'rho_product', 'wavelength_cos_sza',
        'wavelength_cos_vza'
    ]


# ==================== Data Loader ====================
class SimpleDataLoader:
    """Simplified data loader"""

    def __init__(self, config):
        self.config = config
        self.station_coords = None

    def load_stations(self):
        """Load station coordinates"""
        import netCDF4 as nc

        with nc.Dataset(self.config.DATA_PATHS['luts']) as ds:
            # Read stations
            stations_raw = ds.variables['Station'][:]
            stations = []
            for s in stations_raw:
                if isinstance(s, bytes):
                    s = s.decode('utf-8')
                stations.append(str(s).strip())

            # Read coordinates
            lats = ds.variables['Lat'][:]
            lons = ds.variables['Lon'][:]

            if isinstance(lats, np.ma.MaskedArray):
                lats = lats.filled(np.nan)
            if isinstance(lons, np.ma.MaskedArray):
                lons = lons.filled(np.nan)

            df = pd.DataFrame({
                'station': stations,
                'lat': lats,
                'lon': lons
            }).dropna(subset=['lat', 'lon']).drop_duplicates('station')

        self.station_coords = df
        return df

    def select_stations(self, n_stations=50):
        """Select stations"""
        stations_df = self.load_stations()

        if len(stations_df) <= n_stations:
            return stations_df['station'].tolist()

        # Evenly select
        return stations_df.sample(n=n_stations, random_state=42)['station'].tolist()

    def load_single_hour(self, date_str, hour, stations):
        """Load single hour data"""
        import netCDF4 as nc
        import os

        hour_str = f"{hour * 100:04d}"
        file_path = os.path.join(
            self.config.DATA_PATHS['hourly_sozSR'],
            date_str[:4],
            date_str[4:6],
            f"H8_hourly_sozSR_angles_{date_str}_{hour_str}.nc"
        )

        if not os.path.exists(file_path):
            return pd.DataFrame()

        with nc.Dataset(file_path) as ds:
            # Read station list
            file_stations_raw = ds.variables['Station'][:]
            file_stations = []
            for s in file_stations_raw:
                if isinstance(s, bytes):
                    s = s.decode('utf-8')
                file_stations.append(str(s).strip())

            # Filter required stations
            station_indices = []
            valid_stations = []
            for station in stations:
                if station in file_stations:
                    idx = file_stations.index(station)
                    station_indices.append(idx)
                    valid_stations.append(station)

            if not station_indices:
                return pd.DataFrame()

            # Extract data
            data_dict = {'station': valid_stations}

            # Angle data
            angle_cols = ['SOZ', 'SAZ', 'SOA', 'SAA']
            for col in angle_cols:
                if col in ds.variables:
                    var_data = ds.variables[col][:][station_indices]
                    if isinstance(var_data, np.ma.MaskedArray):
                        var_data = var_data.filled(np.nan)
                    data_dict[col] = var_data

            # Band data (1-6)
            for band in ['01', '02', '03', '04', '05', '06']:
                col = f'Albedo_{band}'
                if col in ds.variables:
                    var_data = ds.variables[col][:][station_indices]
                    if isinstance(var_data, np.ma.MaskedArray):
                        var_data = var_data.filled(np.nan)
                    data_dict[f'TOA_Albedo_{band}'] = var_data / 100.0

            df = pd.DataFrame(data_dict)

            if df.empty:
                return df

            # Add time
            dt = datetime.strptime(date_str, "%Y%m%d") + timedelta(hours=hour)
            df['datetime'] = dt

            return df


# ==================== Feature Preparer ====================
class FeaturePreparer:
    """Feature preparer"""

    def __init__(self, config):
        self.config = config

    def calculate_physical_features(self, sza, vza, saa, soa):
        """Calculate physical features"""
        # Relative azimuth angle
        raa = np.abs(soa - saa)

        # Convert angles to radians
        sza_rad = np.radians(sza)
        vza_rad = np.radians(vza)
        raa_rad = np.radians(raa)

        features = {
            'sza': sza,
            'vza': vza,
            'raa': raa,
            'cos_sza': np.cos(sza_rad),
            'sin_sza': np.sin(sza_rad),
            'cos_vza': np.cos(vza_rad),
            'sin_vza': np.sin(vza_rad),
            'cos_raa': np.cos(raa_rad),
            'sin_raa': np.sin(raa_rad)
        }

        # Scattering angle
        cos_scat = -np.cos(sza_rad) * np.cos(vza_rad) + \
                   np.sin(sza_rad) * np.sin(vza_rad) * np.cos(raa_rad)
        cos_scat = np.clip(cos_scat, -1.0, 1.0)
        features['scattering_angle'] = np.degrees(np.arccos(cos_scat))

        # Air mass
        cos_sza_safe = np.clip(features['cos_sza'], 0.001, 1.0)
        cos_vza_safe = np.clip(features['cos_vza'], 0.001, 1.0)
        features['airmass_sza'] = 1.0 / cos_sza_safe
        features['airmass_vza'] = 1.0 / cos_vza_safe
        features['total_airmass'] = features['airmass_sza'] + features['airmass_vza']

        return features

    def prepare_features_for_band(self, data_df, band_id):
        """Prepare features for a single band"""
        df = data_df.copy()

        # Ensure necessary columns exist
        if 'SOZ' not in df.columns or f'TOA_Albedo_{band_id}' not in df.columns:
            return pd.DataFrame()

        # Basic features
        features_df = pd.DataFrame(index=df.index)

        # Geometric features
        physical_features = self.calculate_physical_features(
            df['SOZ'].values,
            df['SAZ'].values,
            df['SOA'].values,
            df['SAA'].values
        )

        for key, value in physical_features.items():
            features_df[key] = value

        # Atmospheric parameters (simplified: using default values)
        features_df['aod550'] = df.get('AOD550', 0.1)
        features_df['h2o'] = df.get('water', 2.0)
        features_df['o3'] = df.get('ozone', 0.3)

        # Wavelength
        features_df['wavelength'] = self.config.BAND_CONFIG[band_id]['wavelength']

        # Reflectance features
        toa_col = f'TOA_Albedo_{band_id}'
        features_df['rho_toa'] = df[toa_col]
        features_df['rho_retrieved'] = df[toa_col]  # Assume TOA is the retrieval result

        # Derived features
        features_df['vza_sza_ratio'] = features_df['vza'] / (features_df['sza'] + 1e-6)
        features_df['vza_minus_sza'] = features_df['vza'] - features_df['sza']
        features_df['vza_plus_sza'] = features_df['vza'] + features_df['sza']

        # Interaction features
        features_df['aod_airmass'] = features_df['aod550'] * features_df['total_airmass']
        features_df['aod_wavelength'] = features_df['aod550'] / features_df['wavelength']

        if 'rho_retrieved' in features_df.columns:
            features_df['rho_ratio'] = features_df['rho_toa'] / (features_df['rho_retrieved'] + 1e-6)
            features_df['rho_diff'] = features_df['rho_toa'] - features_df['rho_retrieved']
            features_df['rho_product'] = features_df['rho_toa'] * features_df['rho_retrieved']

        features_df['wavelength_cos_sza'] = features_df['wavelength'] * features_df['cos_sza']
        features_df['wavelength_cos_vza'] = features_df['wavelength'] * features_df['cos_vza']

        # Add metadata
        features_df['original_index'] = df.index
        features_df['station'] = df['station'] if 'station' in df.columns else 'unknown'
        features_df['datetime'] = df['datetime'] if 'datetime' in df.columns else None
        features_df['band'] = band_id

        return features_df


# ==================== Validation Workflow ====================
class ExternalValidator:
    """External validator"""

    def __init__(self, config):
        self.config = config
        self.data_loader = SimpleDataLoader(config)
        self.feature_preparer = FeaturePreparer(config)

    def load_validation_data(self, n_stations=50, start_date="20160101", end_date="20160110"):
        """Load validation data"""
        print(f"Loading validation data: {start_date} to {end_date}, {n_stations} stations")

        # Select stations
        stations = self.data_loader.select_stations(n_stations)
        print(f"Selected stations: {stations[:5]}...")

        # Generate date range
        start_dt = datetime.strptime(start_date, "%Y%m%d")
        end_dt = datetime.strptime(end_date, "%Y%m%d")

        all_data = []

        current_dt = start_dt
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y%m%d")
            print(f"Processing date: {date_str}")

            # Daytime hours (8-16)
            for hour in range(8, 17):
                df_hour = self.data_loader.load_single_hour(date_str, hour, stations)

                if not df_hour.empty:
                    # Filter data with SOZ too large
                    df_hour = df_hour[df_hour['SOZ'] <= 85].copy()

                    # Add default atmospheric parameters (if not exist)
                    if 'AOD550' not in df_hour.columns:
                        df_hour['AOD550'] = 0.1
                    if 'water' not in df_hour.columns:
                        df_hour['water'] = 2.0
                    if 'ozone' not in df_hour.columns:
                        df_hour['ozone'] = 0.3

                    all_data.append(df_hour)

            current_dt += timedelta(days=1)

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            print(f"Data loading completed: {len(final_df)} records")
            print(f"Number of stations: {final_df['station'].nunique()}")
            return final_df

        return pd.DataFrame()

    def predict_correction(self, data_df, model_path):
        """Predict correction"""
        print(f"Loading model: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        all_predictions = []

        # Predict for each band
        for band_id in ['01', '02', '03', '04', '05', '06']:
            print(f"Processing band {band_id}")

            # Prepare features
            features_df = self.feature_preparer.prepare_features_for_band(data_df, band_id)

            if features_df.empty:
                print(f"Band {band_id} has no data")
                continue

            # Ensure feature order
            feature_cols = []
            missing_features = []

            for feat in self.config.EXPECTED_FEATURES:
                if feat in features_df.columns:
                    feature_cols.append(feat)
                else:
                    missing_features.append(feat)
                    # Fill default values
                    if feat in ['h2o', 'o3', 'aod550']:
                        features_df[feat] = 0.0
                    elif 'rho' in feat:
                        features_df[feat] = 0.1
                    else:
                        features_df[feat] = 0.0
                    feature_cols.append(feat)

            if missing_features:
                print(f"  Filling missing features: {len(missing_features)} features")

            # Predict
            X = features_df[self.config.EXPECTED_FEATURES]
            predictions = model.predict(X)

            # Save results
            band_results = pd.DataFrame({
                'original_index': features_df['original_index'].values,
                'band': band_id,
                'predicted_correction': predictions,
                'rho_TOA': features_df['rho_toa'].values,
                'station': features_df['station'].values,
                'datetime': features_df['datetime'].values
            })

            # Calculate corrected reflectance
            band_results['rho_true_estimated'] = band_results['rho_TOA'] - band_results['predicted_correction']
            band_results['rho_true_estimated'] = band_results['rho_true_estimated'].clip(0, 1)

            all_predictions.append(band_results)

            # Statistics
            print(f"  Correction: mean={band_results['predicted_correction'].mean():.6f}, "
                  f"range=[{band_results['predicted_correction'].min():.6f}, "
                  f"{band_results['predicted_correction'].max():.6f}]")

        if all_predictions:
            return pd.concat(all_predictions, ignore_index=True)
        return pd.DataFrame()

    def apply_correction(self, original_data, predictions):
        """Apply correction to original data"""
        corrected_data = original_data.copy()

        # Initialize correction result columns
        for band_id in ['01', '02', '03', '04', '05', '06']:
            corrected_data[f'correction_{band_id}'] = np.nan
            corrected_data[f'corrected_{band_id}'] = np.nan

        # Map correction results
        for _, pred_row in predictions.iterrows():
            orig_idx = int(pred_row['original_index'])
            band_id = pred_row['band']

            if orig_idx < len(corrected_data):
                correction = pred_row['predicted_correction']
                corrected_data.at[orig_idx, f'correction_{band_id}'] = correction

                toa_col = f'TOA_Albedo_{band_id}'
                if toa_col in corrected_data.columns:
                    corrected_data.at[orig_idx, f'corrected_{band_id}'] = (
                            corrected_data.at[orig_idx, toa_col] - correction
                    )

        return corrected_data

    def generate_report(self, corrected_data, predictions, output_dir):
        """Generate report"""
        import matplotlib.pyplot as plt

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save data
        data_path = output_dir / "corrected_data.parquet"
        corrected_data.to_parquet(data_path)
        print(f"Corrected data saved: {data_path}")

        # 2. Band statistics
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("EXTERNAL VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total samples: {len(corrected_data)}")
        report_lines.append(f"Stations: {corrected_data['station'].nunique()}")

        for band_id in ['01', '02', '03', '04', '05', '06']:
            corr_col = f'correction_{band_id}'
            if corr_col in corrected_data.columns:
                corr_data = corrected_data[corr_col].dropna()
                if len(corr_data) > 0:
                    band_mask = predictions['band'] == band_id
                    band_pred = predictions[band_mask]

                    orig_var = band_pred['rho_TOA'].var() if len(band_pred) > 1 else 0
                    corr_var = band_pred['rho_true_estimated'].var() if len(band_pred) > 1 else 0
                    var_reduction = (orig_var - corr_var) / orig_var * 100 if orig_var > 0 else 0

                    report_lines.append(f"\nBand {band_id}:")
                    report_lines.append(f"  Samples: {len(corr_data)}")
                    report_lines.append(f"  Correction mean: {corr_data.mean():.6f}")
                    report_lines.append(f"  Correction std: {corr_data.std():.6f}")
                    report_lines.append(f"  Positive correction: {(corr_data > 0).sum() / len(corr_data) * 100:.1f}%")
                    report_lines.append(f"  Original reflectance variance: {orig_var:.6f}")
                    report_lines.append(f"  Corrected reflectance variance: {corr_var:.6f}")
                    report_lines.append(f"  Variance reduction: {var_reduction:.1f}%")

        # 3. Save report
        report_path = output_dir / "validation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        print(f"Report saved: {report_path}")

        # 4. Generate simple plots
        self.generate_plots(predictions, corrected_data, output_dir)

        return report_path

    def generate_plots(self, predictions, corrected_data, output_dir):
        """Generate plots with scientific style"""
        import matplotlib.pyplot as plt

        # Set scientific plotting style (RSE journal style)
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
            'mathtext.fontset': 'custom',
            'mathtext.rm': 'Arial',
            'mathtext.it': 'Arial:italic',
            'mathtext.bf': 'Arial:bold',
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 11,
            'figure.dpi': 300,
        })

        # Figure 1: Correction distribution
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, band_id in enumerate(['01', '02', '03', '04', '05', '06']):
            if idx >= len(axes):
                break

            band_mask = predictions['band'] == band_id
            band_data = predictions[band_mask]

            if len(band_data) > 0:
                axes[idx].hist(band_data['predicted_correction'], bins=20, alpha=0.7,
                               color='steelblue', edgecolor='black')
                axes[idx].axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                axes[idx].set_xlabel('Correction Value', fontsize=9)
                axes[idx].set_ylabel('Frequency', fontsize=9)
                axes[idx].set_title(f'Band {band_id}', fontsize=10, fontweight='bold')
                axes[idx].grid(True, alpha=0.3, linestyle='--')
                axes[idx].spines['top'].set_visible(False)
                axes[idx].spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / 'correction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Figure 2: SZA vs Reflectance
        if 'SOZ' in corrected_data.columns:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            for idx, band_id in enumerate(['01', '02', '03', '04', '05', '06']):
                if idx >= len(axes):
                    break

                corr_col = f'corrected_{band_id}'
                toa_col = f'TOA_Albedo_{band_id}'

                if corr_col in corrected_data.columns and toa_col in corrected_data.columns:
                    valid_mask = corrected_data[corr_col].notna() & corrected_data['SOZ'].notna()
                    if valid_mask.sum() > 0:
                        # Original reflectance
                        axes[idx].scatter(corrected_data.loc[valid_mask, 'SOZ'],
                                          corrected_data.loc[valid_mask, toa_col],
                                          alpha=0.5, s=15, label='Original TOA',
                                          color='tab:blue', marker='o', edgecolors='white', linewidth=0.5)
                        # Corrected reflectance
                        axes[idx].scatter(corrected_data.loc[valid_mask, 'SOZ'],
                                          corrected_data.loc[valid_mask, corr_col],
                                          alpha=0.5, s=15, label='Corrected',
                                          color='tab:red', marker='s', edgecolors='white', linewidth=0.5)

                        axes[idx].set_xlabel('Solar Zenith Angle (°)', fontsize=9)
                        axes[idx].set_ylabel('Reflectance', fontsize=9)
                        axes[idx].set_title(f'Band {band_id}: SZA Dependence', fontsize=10, fontweight='bold')
                        axes[idx].legend(fontsize=8, framealpha=0.9)
                        axes[idx].grid(True, alpha=0.3, linestyle='--')
                        axes[idx].spines['top'].set_visible(False)
                        axes[idx].spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig(output_dir / 'sza_dependence.png', dpi=300, bbox_inches='tight')
            plt.close()


# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(description='External Validation Workflow')
    parser.add_argument('--stations', type=int, default=50,
                        help='Number of stations (default: 50)')
    parser.add_argument('--start_date', type=str, default='20160101',
                        help='Start date YYYYMMDD (default: 20160101)')
    parser.add_argument('--end_date', type=str, default='20160110',
                        help='End date YYYYMMDD (default: 20160110)')
    parser.add_argument('--model_path', type=str,
                        default=r'D:\6S_Self_Consistency_Zenith_Angles_v0.4\models\refactored\model_training_20260117_152723\models\XGBoost_model.pkl',
                        help='Model path')
    parser.add_argument('--output_dir', type=str, default='./validation_results',
                        help='Output directory (default: ./validation_results)')
    parser.add_argument('--use_existing', type=str, default=None,
                        help='Path to existing data file (skip data loading)')

    args = parser.parse_args()

    print("=" * 70)
    print("EXTERNAL VALIDATION WORKFLOW")
    print("=" * 70)
    print(f"Stations: {args.stations}")
    print(f"Time period: {args.start_date} to {args.end_date}")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 70)

    # Create validator
    config = Config()
    validator = ExternalValidator(config)

    # Load data
    if args.use_existing and Path(args.use_existing).exists():
        print(f"Using existing data: {args.use_existing}")
        data_df = pd.read_parquet(args.use_existing)
    else:
        data_df = validator.load_validation_data(
            n_stations=args.stations,
            start_date=args.start_date,
            end_date=args.end_date
        )

        if data_df.empty:
            print("Error: Unable to load data")
            return 1

        # Save raw data
        raw_path = Path(args.output_dir) / "raw_data.parquet"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        data_df.to_parquet(raw_path)
        print(f"Raw data saved: {raw_path}")

    print(f"Data shape: {data_df.shape}")

    # Predict corrections
    predictions = validator.predict_correction(data_df, args.model_path)

    if predictions.empty:
        print("Error: Prediction failed")
        return 1

    print(f"Prediction completed: {len(predictions)} samples")

    # Apply corrections
    corrected_data = validator.apply_correction(data_df, predictions)

    # Generate report
    report_path = validator.generate_report(corrected_data, predictions, args.output_dir)

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETED!")
    print("=" * 70)
    print(f"Report: {report_path}")
    print(f"Data: {Path(args.output_dir) / 'corrected_data.parquet'}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    main()