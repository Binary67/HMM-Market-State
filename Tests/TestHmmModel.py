import unittest
import numpy as np
import pandas as pd
from HmmTrading.HmmModel import TradingHmm

class TestTradingHmm(unittest.TestCase):

    def setUp(self):
        """Setup common data for tests."""
        print("Setting up data for TestTradingHmm...")
        # Generate more data points to avoid issues with small sample sizes for HMM
        ClosePrices = np.cumprod(1 + np.random.normal(0.0001, 0.015, 300)) * 150
        self.DummyPriceData = pd.DataFrame({'Close': ClosePrices})
        self.DummyPriceData.index = pd.date_range(start='2022-01-01', periods=300)

        self.HmmModel = TradingHmm(NumberOfHiddenStates=2, MaxIterations=20, Tolerance=0.01) # Faster training for tests
        print("Setup complete.")

    def test_Initialization_ValidParameters(self):
        print("Running test_Initialization_ValidParameters...")
        Model = TradingHmm(NumberOfHiddenStates=3, MaxIterations=50, Tolerance=1e-3)
        self.assertEqual(Model.NumberOfHiddenStates, 3)
        self.assertEqual(Model.MaxIterations, 50)
        self.assertEqual(Model.Tolerance, 1e-3)
        self.assertFalse(Model.IsTrained)
        print("test_Initialization_ValidParameters PASSED")

    def test_Initialization_InvalidParameters(self):
        print("Running test_Initialization_InvalidParameters...")
        with self.assertRaises(ValueError):
            TradingHmm(NumberOfHiddenStates=0)
        with self.assertRaises(ValueError):
            TradingHmm(MaxIterations=-5)
        with self.assertRaises(ValueError):
            TradingHmm(Tolerance=0)
        print("test_Initialization_InvalidParameters PASSED")

    def test_PreprocessData_ValidData(self):
        print("Running test_PreprocessData_ValidData...")
        Processed = self.HmmModel.PreprocessData(self.DummyPriceData)
        self.assertIsInstance(Processed, np.ndarray)
        self.assertEqual(Processed.ndim, 2)
        self.assertEqual(Processed.shape[1], 2) # LogReturns and Volatility
        # Expected length: len(self.DummyPriceData) - WindowSize (5 for volatility)
        self.assertEqual(Processed.shape[0], len(self.DummyPriceData) - 5)
        self.assertFalse(np.isnan(Processed).any())
        print(f"test_PreprocessData_ValidData PASSED - Processed shape: {Processed.shape}")

    def test_PreprocessData_EmptyDataFrame(self):
        print("Running test_PreprocessData_EmptyDataFrame...")
        with self.assertRaises(ValueError):
            self.HmmModel.PreprocessData(pd.DataFrame())
        print("test_PreprocessData_EmptyDataFrame PASSED")

    def test_PreprocessData_MissingCloseColumn(self):
        print("Running test_PreprocessData_MissingCloseColumn...")
        InvalidData = pd.DataFrame({'Open': self.DummyPriceData['Close']})
        with self.assertRaises(ValueError):
            self.HmmModel.PreprocessData(InvalidData)
        print("test_PreprocessData_MissingCloseColumn PASSED")

    def test_PreprocessData_TooShortData(self):
        print("Running test_PreprocessData_TooShortData...")
        # Data with 5 rows: LogReturns (4 rows), Volatility.rolling(5) on 4 rows -> all NaN -> empty after dropna
        ShortData = pd.DataFrame({'Close': [100, 101, 102, 103, 104]})
        with self.assertRaises(ValueError) as Context: # Expecting "Preprocessed features are empty"
            self.HmmModel.PreprocessData(ShortData)
        self.assertIn("Preprocessed features are empty", str(Context.exception))
        print("test_PreprocessData_TooShortData PASSED")


    def test_Train_ValidData(self):
        print("Running test_Train_ValidData...")
        Processed = self.HmmModel.PreprocessData(self.DummyPriceData)
        self.HmmModel.Train(ProcessedData=Processed)
        self.assertTrue(self.HmmModel.IsTrained)
        self.assertIsNotNone(self.HmmModel.Model.means_)
        print("test_Train_ValidData PASSED")

    def test_Train_EmptyData(self):
        print("Running test_Train_EmptyData...")
        EmptyProcessedData = np.array([]).reshape(0, 2)
        with self.assertRaises(ValueError):
            self.HmmModel.Train(ProcessedData=EmptyProcessedData)
        self.assertFalse(self.HmmModel.IsTrained)
        print("test_Train_EmptyData PASSED")

    def test_PredictStates_BeforeTraining(self):
        print("Running test_PredictStates_BeforeTraining...")
        Processed = self.HmmModel.PreprocessData(self.DummyPriceData)
        with self.assertRaises(RuntimeError):
            self.HmmModel.PredictStates(Data=Processed)
        print("test_PredictStates_BeforeTraining PASSED")

    def test_PredictStates_AfterTraining(self):
        print("Running test_PredictStates_AfterTraining...")
        Processed = self.HmmModel.PreprocessData(self.DummyPriceData)
        self.HmmModel.Train(ProcessedData=Processed)
        Predicted = self.HmmModel.PredictStates(Data=Processed)
        self.assertIsInstance(Predicted, np.ndarray)
        self.assertEqual(len(Predicted), len(Processed))
        self.assertTrue(all(State >= 0 and State < self.HmmModel.NumberOfHiddenStates for State in Predicted))
        print(f"test_PredictStates_AfterTraining PASSED - Predicted states shape: {Predicted.shape}")

    def test_PredictStates_EmptyData(self):
        print("Running test_PredictStates_EmptyData...")
        Processed = self.HmmModel.PreprocessData(self.DummyPriceData)
        self.HmmModel.Train(ProcessedData=Processed)
        EmptyDataForPrediction = np.array([]).reshape(0,2)
        with self.assertRaises(ValueError): # hmmlearn's predict might also raise if data is empty after checks
            self.HmmModel.PredictStates(Data=EmptyDataForPrediction)
        print("test_PredictStates_EmptyData PASSED")


    def test_GenerateTradingSignals_ValidStates(self):
        print("Running test_GenerateTradingSignals_ValidStates...")
        # Example states: 0, 0, 1, 1, 0, 1 -> signals: 0, 0, 1, 0, -1, 1
        PredictedStates = np.array([0, 0, 1, 1, 0, 1])
        Signals = self.HmmModel.GenerateTradingSignals(PredictedStates=PredictedStates)
        ExpectedSignals = pd.Series([0, 0, 1, 0, -1, 1], dtype=int)
        pd.testing.assert_series_equal(Signals, ExpectedSignals, check_names=False)
        print("test_GenerateTradingSignals_ValidStates PASSED")

    def test_GenerateTradingSignals_EmptyStates(self):
        print("Running test_GenerateTradingSignals_EmptyStates...")
        PredictedStates = np.array([])
        Signals = self.HmmModel.GenerateTradingSignals(PredictedStates=PredictedStates)
        self.assertIsInstance(Signals, pd.Series)
        self.assertTrue(Signals.empty)
        self.assertEqual(Signals.dtype, int)
        print("test_GenerateTradingSignals_EmptyStates PASSED")

    def test_GenerateTradingSignals_SingleState(self):
        print("Running test_GenerateTradingSignals_SingleState...")
        PredictedStates = np.array([0]) # Single state
        Signals = self.HmmModel.GenerateTradingSignals(PredictedStates=PredictedStates)
        ExpectedSignals = pd.Series([0], dtype=int)
        pd.testing.assert_series_equal(Signals, ExpectedSignals, check_names=False)
        print("test_GenerateTradingSignals_SingleState PASSED")

    def test_FullWorkflow(self):
        print("Running test_FullWorkflow...")
        # 1. Preprocess
        ProcessedFeatures = self.HmmModel.PreprocessData(PriceData=self.DummyPriceData)
        self.assertTrue(ProcessedFeatures.shape[0] > 0)

        # 2. Train
        self.HmmModel.Train(ProcessedData=ProcessedFeatures)
        self.assertTrue(self.HmmModel.IsTrained)

        # 3. Predict
        PredictedHiddenStates = self.HmmModel.PredictStates(Data=ProcessedFeatures)
        self.assertEqual(len(PredictedHiddenStates), ProcessedFeatures.shape[0])

        # 4. Generate Signals
        TradingSignals = self.HmmModel.GenerateTradingSignals(PredictedStates=PredictedHiddenStates)
        self.assertEqual(len(TradingSignals), len(PredictedHiddenStates))
        print("test_FullWorkflow PASSED")


if __name__ == '__main__':
    print("Starting tests for HmmModel...")
    # Running with verbosity to see individual test names and print statements
    unittest.main(argv=['first-arg-is-ignored'], exit=False, verbosity=0)
    print("Finished tests for HmmModel.")
