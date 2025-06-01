import numpy as np
import pandas as pd
from hmmlearn import hmm

class TradingHmm:
    def __init__(self, NumberOfHiddenStates: int = 2, MaxIterations: int = 100, Tolerance: float = 1e-2):
        if not isinstance(NumberOfHiddenStates, int) or NumberOfHiddenStates <= 0:
            raise ValueError("NumberOfHiddenStates must be a positive integer.")
        if not isinstance(MaxIterations, int) or MaxIterations <= 0:
            raise ValueError("MaxIterations must be a positive integer.")
        if not isinstance(Tolerance, float) or Tolerance <= 0:
            raise ValueError("Tolerance must be a positive float.")

        self.NumberOfHiddenStates = NumberOfHiddenStates
        self.MaxIterations = MaxIterations
        self.Tolerance = Tolerance
        self.Model = hmm.GaussianHMM(
            n_components=self.NumberOfHiddenStates,
            covariance_type="diag",
            n_iter=self.MaxIterations,
            tol=self.Tolerance,
            random_state=42
        )
        self.IsTrained = False

    def PreprocessData(self, PriceData: pd.DataFrame) -> np.ndarray:
        if not isinstance(PriceData, pd.DataFrame) or PriceData.empty:
            raise ValueError("PriceData must be a non-empty pandas DataFrame.")
        if 'Close' not in PriceData.columns:
            raise ValueError("PriceData must contain a 'Close' column.")

        LogReturns = np.log(PriceData['Close'] / PriceData['Close'].shift(1))
        WindowSize = 5
        Volatility = LogReturns.rolling(window=WindowSize).std()

        Features = pd.concat([LogReturns, Volatility], axis=1)
        Features.columns = ['LogReturns', 'Volatility']
        Features = Features.dropna()

        if Features.empty:
            raise ValueError("Preprocessed features are empty. Input data might be too short or contain too many NaNs.")

        return Features.values

    def Train(self, ProcessedData: np.ndarray):
        if not isinstance(ProcessedData, np.ndarray) or ProcessedData.ndim != 2:
            raise ValueError("ProcessedData must be a 2D NumPy array.")
        if ProcessedData.shape[0] == 0:
            raise ValueError("ProcessedData is empty, cannot train the model.")

        print(f"Training HMM with {self.NumberOfHiddenStates} hidden states...")
        try:
            self.Model.fit(ProcessedData)
            self.IsTrained = True
            print("HMM training completed successfully.")
        except Exception as E:
            print(f"An error occurred during HMM training: {E}")
            self.IsTrained = False
            raise E

    def PredictStates(self, Data: np.ndarray) -> np.ndarray:
        if not self.IsTrained:
            raise RuntimeError("Model has not been trained yet. Call Train() first.")
        if not isinstance(Data, np.ndarray) or Data.ndim != 2:
            raise ValueError("Data for prediction must be a 2D NumPy array.")
        if Data.shape[0] == 0:
            raise ValueError("Data for prediction is empty.")

        print("Predicting hidden states...")
        try:
            PredictedStates = self.Model.predict(Data)
            print("Hidden states prediction completed.")
            return PredictedStates
        except Exception as E:
            print(f"An error occurred during state prediction: {E}")
            raise E

    def GenerateTradingSignals(self, PredictedStates: np.ndarray) -> pd.Series:
        if not isinstance(PredictedStates, np.ndarray) or PredictedStates.ndim != 1:
            raise ValueError("PredictedStates must be a 1D NumPy array.")
        if PredictedStates.size == 0:
            # Return an empty Series with integer dtype if no states
            return pd.Series([], dtype=int, index=pd.Index([]))


        TradingSignals = pd.Series(index=range(len(PredictedStates)), dtype=int)
        if len(PredictedStates) > 0: # Ensure there are states to process
            TradingSignals.iloc[0] = 0

        for I in range(1, len(PredictedStates)):
            CurrentState = PredictedStates[I]
            PreviousState = PredictedStates[I-1]

            if CurrentState > PreviousState:
                TradingSignals.iloc[I] = 1
            elif CurrentState < PreviousState:
                TradingSignals.iloc[I] = -1
            else:
                TradingSignals.iloc[I] = 0

        print("Trading signals generated.")
        return TradingSignals.astype(np.int64) # Explicitly cast to ensure int64 dtype

if __name__ == '__main__':
    DummyPriceData = pd.DataFrame({
        'Close': np.cumprod(1 + np.random.normal(0.0001, 0.01, 200)) * 100
    })
    DummyPriceData.index = pd.date_range(start='2022-01-01', periods=200)

    print("--- Testing TradingHmm ---")
    try:
        MyHmmModel = TradingHmm(NumberOfHiddenStates=2)

        print("\n--- Preprocessing Data ---")
        ProcessedFeatures = MyHmmModel.PreprocessData(PriceData=DummyPriceData)
        print(f"Shape of processed features: {ProcessedFeatures.shape}")
        # print(f"First 5 rows of processed features:\n{ProcessedFeatures[:5]}") # Commented out for brevity in logs

        print("\n--- Training HMM ---")
        MyHmmModel.Train(ProcessedData=ProcessedFeatures)
        if MyHmmModel.IsTrained:
            print("Model is trained.")
            # print(f"Model Means:\n{MyHmmModel.Model.means_}") # Commented out
            # print(f"Model Covariances:\n{MyHmmModel.Model.covars_}") # Commented out
            # print(f"Model Transition Matrix:\n{MyHmmModel.Model.transmat_}") # Commented out

            print("\n--- Predicting States ---")
            PredictedHiddenStates = MyHmmModel.PredictStates(Data=ProcessedFeatures)
            # print(f"Predicted states (first 20): {PredictedHiddenStates[:20]}") # Commented out

            print("\n--- Generating Trading Signals ---")
            Signals = MyHmmModel.GenerateTradingSignals(PredictedStates=PredictedHiddenStates)
            # print(f"Generated signals (first 20):\n{Signals.head(20)}") # Commented out

            print("\n--- Verifying a few signals ---")
            CorrectSignalCount = 0
            TotalSignalsChecked = 0
            for K in range(1, min(20, len(PredictedHiddenStates))):
                TotalSignalsChecked +=1
                Signal = Signals.iloc[K]
                CurrentS = PredictedHiddenStates[K]
                PreviousS = PredictedHiddenStates[K-1]
                ExpectedSignal = 0
                if CurrentS > PreviousS: ExpectedSignal = 1
                elif CurrentS < PreviousS: ExpectedSignal = -1
                # print(f"Index {K}: PrevState={PreviousS}, CurrState={CurrentS}, Signal={Signal}, Expected={ExpectedSignal} -> {'Correct' if Signal == ExpectedSignal else 'INCORRECT'}")
                if Signal == ExpectedSignal:
                    CorrectSignalCount +=1
            print(f"Signal verification: {CorrectSignalCount}/{TotalSignalsChecked} correct for the first {min(20, len(PredictedHiddenStates))} signals checked.")


    except ValueError as VE:
        print(f"ValueError in example usage: {VE}")
    except RuntimeError as RE:
        print(f"RuntimeError in example usage: {RE}")
    except Exception as Ex:
        print(f"An unexpected error occurred in example usage: {Ex}")

    print("\n--- Test Error Handling ---")
    try:
        InvalidModel = TradingHmm(NumberOfHiddenStates=0)
    except ValueError as VE:
        print(f"Correctly caught error for invalid NumberOfHiddenStates: {VE}")

    try:
        ModelForTest = TradingHmm()
        ModelForTest.PreprocessData(PriceData=pd.DataFrame())
    except ValueError as VE:
        print(f"Correctly caught error for empty PriceData in PreprocessData: {VE}")

    try:
        ModelForTest = TradingHmm()
        ShortData = pd.DataFrame({'Close': [100, 101, 102]})
        ModelForTest.PreprocessData(PriceData=ShortData)
    except ValueError as VE:
        print(f"Correctly caught error for too short PriceData in PreprocessData: {VE}")

    try:
        ModelForTest = TradingHmm()
        ModelForTest.Train(ProcessedData=np.array([]).reshape(0,2)) # Empty but 2D
    except ValueError as VE:
        print(f"Correctly caught error for empty ProcessedData in Train: {VE}")

    try:
        ModelForTest = TradingHmm()
        ModelForTest.PredictStates(Data=np.array([[0.1, 0.01]]))
    except RuntimeError as RE:
        print(f"Correctly caught error for PredictStates before training: {RE}")
