import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Optional


class FeatureEngineering:
    """
    Dynamic feature engineering class for technical analysis indicators.
    Allows flexible addition of technical indicators with customizable parameters.
    """
    
    def __init__(self):
        self.TechnicalIndicators: Dict[str, Callable] = {
            "SMA": self._CalculateSimpleMovingAverage,
            "EMA": self._CalculateExponentialMovingAverage,
            "RSI": self._CalculateRelativeStrengthIndex,
            "MACD": self._CalculateMovingAverageConvergenceDivergence,
            "BollingerBands": self._CalculateBollingerBands,
            "ATR": self._CalculateAverageTrueRange,
            "Stochastic": self._CalculateStochasticOscillator,
            "OnBalanceVolume": self._CalculateOnBalanceVolume,
            "WilliamsPercentR": self._CalculateWilliamsPercentR,
            "CommodityChannelIndex": self._CalculateCommodityChannelIndex,
        }
    
    def ApplyTechnicalAnalysis(self, Data: pd.DataFrame, IndicatorsToApply: List[str], 
                              IndicatorParameters: Optional[Dict[str, Dict]] = None) -> pd.DataFrame:
        """
        Apply selected technical indicators to the dataframe.
        
        Args:
            Data: DataFrame with OHLCV data
            IndicatorsToApply: List of indicator names to apply
            IndicatorParameters: Dictionary of parameters for each indicator
            
        Returns:
            DataFrame with additional technical indicator columns
        """
        ResultData = Data.copy()
        
        if IndicatorParameters is None:
            IndicatorParameters = {}
        
        for Indicator in IndicatorsToApply:
            if Indicator in self.TechnicalIndicators:
                Parameters = IndicatorParameters.get(Indicator, {})
                ResultData = self.TechnicalIndicators[Indicator](ResultData, **Parameters)
            else:
                print(f"Warning: Indicator '{Indicator}' not found in available indicators")
        
        return ResultData
    
    def AddCustomIndicator(self, IndicatorName: str, IndicatorFunction: Callable):
        """
        Add a custom technical indicator to the available indicators.
        
        Args:
            IndicatorName: Name of the new indicator
            IndicatorFunction: Function that takes DataFrame and returns DataFrame with new columns
        """
        self.TechnicalIndicators[IndicatorName] = IndicatorFunction
    
    def GetAvailableIndicators(self) -> List[str]:
        """Return list of available technical indicators."""
        return list(self.TechnicalIndicators.keys())
    
    def _CalculateSimpleMovingAverage(self, Data: pd.DataFrame, Window: int = 20, 
                                    ColumnName: str = "Close") -> pd.DataFrame:
        """Calculate Simple Moving Average."""
        Data[f"SMA_{Window}"] = Data[ColumnName].rolling(window=Window).mean()
        return Data
    
    def _CalculateExponentialMovingAverage(self, Data: pd.DataFrame, Window: int = 20, 
                                         ColumnName: str = "Close") -> pd.DataFrame:
        """Calculate Exponential Moving Average."""
        Data[f"EMA_{Window}"] = Data[ColumnName].ewm(span=Window).mean()
        return Data
    
    def _CalculateRelativeStrengthIndex(self, Data: pd.DataFrame, Window: int = 14, 
                                      ColumnName: str = "Close") -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        Delta = Data[ColumnName].diff()
        Gain = (Delta.where(Delta > 0, 0)).rolling(window=Window).mean()
        Loss = (-Delta.where(Delta < 0, 0)).rolling(window=Window).mean()
        RS = Gain / Loss
        Data[f"RSI_{Window}"] = 100 - (100 / (1 + RS))
        return Data
    
    def _CalculateMovingAverageConvergenceDivergence(self, Data: pd.DataFrame, 
                                                   FastPeriod: int = 12, SlowPeriod: int = 26, 
                                                   SignalPeriod: int = 9, ColumnName: str = "Close") -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        EmaFast = Data[ColumnName].ewm(span=FastPeriod).mean()
        EmaSlow = Data[ColumnName].ewm(span=SlowPeriod).mean()
        Data["MACD"] = EmaFast - EmaSlow
        Data["MACD_Signal"] = Data["MACD"].ewm(span=SignalPeriod).mean()
        Data["MACD_Histogram"] = Data["MACD"] - Data["MACD_Signal"]
        return Data
    
    def _CalculateBollingerBands(self, Data: pd.DataFrame, Window: int = 20, 
                               StandardDeviations: float = 2.0, ColumnName: str = "Close") -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        SMA = Data[ColumnName].rolling(window=Window).mean()
        STD = Data[ColumnName].rolling(window=Window).std()
        Data[f"BB_Upper_{Window}"] = SMA + (STD * StandardDeviations)
        Data[f"BB_Lower_{Window}"] = SMA - (STD * StandardDeviations)
        Data[f"BB_Middle_{Window}"] = SMA
        return Data
    
    def _CalculateAverageTrueRange(self, Data: pd.DataFrame, Window: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        High = Data["High"]
        Low = Data["Low"]
        Close = Data["Close"]
        
        TR1 = High - Low
        TR2 = np.abs(High - Close.shift())
        TR3 = np.abs(Low - Close.shift())
        
        TrueRange = np.maximum(TR1, np.maximum(TR2, TR3))
        Data[f"ATR_{Window}"] = TrueRange.rolling(window=Window).mean()
        return Data
    
    def _CalculateStochasticOscillator(self, Data: pd.DataFrame, KPeriod: int = 14, 
                                     DPeriod: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        LowestLow = Data["Low"].rolling(window=KPeriod).min()
        HighestHigh = Data["High"].rolling(window=KPeriod).max()
        
        Data[f"Stoch_K_{KPeriod}"] = 100 * ((Data["Close"] - LowestLow) / (HighestHigh - LowestLow))
        Data[f"Stoch_D_{DPeriod}"] = Data[f"Stoch_K_{KPeriod}"].rolling(window=DPeriod).mean()
        return Data

    def _CalculateOnBalanceVolume(self, Data: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        CloseDelta = Data["Close"].diff()
        Direction = np.sign(CloseDelta).fillna(0)
        VolumeAdj = Direction * Data["Volume"]
        Data["OnBalanceVolume"] = VolumeAdj.cumsum().fillna(method="ffill")
        return Data

    def _CalculateWilliamsPercentR(self, Data: pd.DataFrame, Window: int = 14) -> pd.DataFrame:
        """Calculate Williams %R."""
        HighestHigh = Data["High"].rolling(window=Window).max()
        LowestLow = Data["Low"].rolling(window=Window).min()
        Data[f"WilliamsR_{Window}"] = -100 * (
            (HighestHigh - Data["Close"]) / (HighestHigh - LowestLow)
        )
        return Data

    def _CalculateCommodityChannelIndex(self, Data: pd.DataFrame, Window: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index."""
        TypicalPrice = (Data["High"] + Data["Low"] + Data["Close"]) / 3
        MovingAverage = TypicalPrice.rolling(window=Window).mean()
        MeanDeviation = TypicalPrice.rolling(window=Window).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        Data[f"CCI_{Window}"] = (TypicalPrice - MovingAverage) / (0.015 * MeanDeviation)
        return Data