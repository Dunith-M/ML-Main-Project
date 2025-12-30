import logging
import pandas as pd
from abc import ABC, abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass
    
class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, df, columns):
        logging.info(f"\n{'='*60}")
        logging.info(f"OUTLIER DETECTION - IQR METHOD")
        logging.info(f"{'='*60}")
        logging.info(f"Starting IQR outlier detection for columns: {columns}")
        outliers =pd.DataFrame(False, index=df.index, columns=columns)
        
        for col in columns:
            logging.info(f"\n--- Processing column: {col} ---")
            df[col] = df[col].astype(float)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1 
            print(f"IQR for {col}: {IQR}")
            logging.info(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers[col].sum()
            logging.info(f"  ✓ Found {outlier_count} outliers ({outlier_count/len(df)*100:.2f}%) out of bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
        
        total_outliers = outliers.any(axis=1).sum()
        logging.info(f"\n{'='*60}")
        logging.info(f'✓ OUTLIER DETECTION COMPLETE - Total rows with outliers: {total_outliers} ({total_outliers/len(df)*100:.2f}%)')
        logging.info(f"{'='*60}\n")
        return outliers