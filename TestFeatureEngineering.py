import pandas as pd
import numpy as np
from FeatureEngineering import FeatureEngineering


def TestFeatureEngineering():
    """Test the FeatureEngineering class with dummy data."""
    
    DummyData = pd.DataFrame({
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0,
                110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
                120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0],
        "High": [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0,
                112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0, 121.0,
                122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0, 131.0],
        "Low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0,
               109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0,
               119.0, 120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0],
        "Close": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0,
                 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0, 120.0,
                 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0, 128.0, 129.0, 130.0],
        "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900,
                  2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900,
                  3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900]
    })
    
    print("Testing FeatureEngineering class...")
    print(f"Original data shape: {DummyData.shape}")
    
    FeatureEngineer = FeatureEngineering()
    
    print(f"\nAvailable indicators: {FeatureEngineer.GetAvailableIndicators()}")
    
    IndicatorsToTest = ["SMA", "EMA", "RSI", "MACD", "BollingerBands", "ATR", "Stochastic"]
    IndicatorParameters = {
        "SMA": {"Window": 10},
        "EMA": {"Window": 10},
        "RSI": {"Window": 14},
        "MACD": {"FastPeriod": 12, "SlowPeriod": 26, "SignalPeriod": 9},
        "BollingerBands": {"Window": 10, "StandardDeviations": 2.0},
        "ATR": {"Window": 14},
        "Stochastic": {"KPeriod": 14, "DPeriod": 3}
    }
    
    EnhancedData = FeatureEngineer.ApplyTechnicalAnalysis(
        DummyData, IndicatorsToTest, IndicatorParameters
    )
    
    print(f"\nEnhanced data shape: {EnhancedData.shape}")
    print(f"New columns added: {EnhancedData.shape[1] - DummyData.shape[1]}")
    
    print("\nNew technical indicator columns:")
    NewColumns = [col for col in EnhancedData.columns if col not in DummyData.columns]
    for col in NewColumns:
        print(f"  - {col}")
    
    print("\nSample of enhanced data (last 5 rows):")
    print(EnhancedData.tail())
    
    print("\nTesting custom indicator addition...")
    def CustomIndicator(Data: pd.DataFrame, Window: int = 5) -> pd.DataFrame:
        """Custom simple price momentum indicator."""
        Data[f"Custom_Momentum_{Window}"] = Data["Close"].pct_change(Window) * 100
        return Data
    
    FeatureEngineer.AddCustomIndicator("CustomMomentum", CustomIndicator)
    
    FinalData = FeatureEngineer.ApplyTechnicalAnalysis(
        EnhancedData, ["CustomMomentum"], {"CustomMomentum": {"Window": 5}}
    )
    
    print(f"\nFinal data shape after custom indicator: {FinalData.shape}")
    print(f"Custom momentum column added: {'Custom_Momentum_5' in FinalData.columns}")
    
    print("\nAll tests completed successfully!")
    return True


if __name__ == "__main__":
    TestFeatureEngineering()