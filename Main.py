import pandas as pd
import matplotlib.pyplot as plt
from HmmTrading.DataDownloader import YahooFinanceDownloader
from HmmTrading.HmmModel import TradingHmm

# Ensure matplotlib uses a non-interactive backend for scripts
import matplotlib
matplotlib.use('Agg')

def Main():
    print("--- HMM Trading Demonstration ---")

    # --- Parameters ---
    TickerSymbol = "AAPL"  # Example stock ticker
    StartDate = "2020-01-01"
    EndDate = "2024-01-01"
    NumberOfHiddenStates = 3  # e.g., Bearish, Neutral, Bullish
    # This should match the WindowSize used in TradingHmm.PreprocessData for LogReturns.rolling(window=WindowSize).std()
    # Plus 1 for the initial drop due to LogReturns = np.log(PriceData['Close'] / PriceData['Close'].shift(1))
    # So, if WindowSize in PreprocessData is 5, AlignmentWindow = 5.
    AlignmentWindow = 5

    # --- 1. Download Data ---
    print(f"\n--- Step 1: Downloading Data for {TickerSymbol} ---")
    DataDownloaderInstance = YahooFinanceDownloader(TickerSymbol=TickerSymbol)
    HistoricalPriceData = DataDownloaderInstance.DownloadData(StartDate=StartDate, EndDate=EndDate)

    if HistoricalPriceData.empty:
        print(f"Failed to download data for {TickerSymbol}. Exiting.")
        return

    print(f"Successfully downloaded {len(HistoricalPriceData)} data points for {TickerSymbol}.")

    # --- 2. Initialize and Preprocess Data for HMM ---
    print("\n--- Step 2: Initializing HMM and Preprocessing Data ---")
    HmmTrader = TradingHmm(NumberOfHiddenStates=NumberOfHiddenStates)

    try:
        ProcessedFeatures = HmmTrader.PreprocessData(PriceData=HistoricalPriceData)
        print(f"Data preprocessed into {ProcessedFeatures.shape[0]} feature vectors with {ProcessedFeatures.shape[1]} features each.")
    except ValueError as VE:
        print(f"Error during data preprocessing: {VE}. Exiting.")
        return

    if ProcessedFeatures.shape[0] == 0:
        print("No features available after preprocessing. Cannot proceed with HMM training. Exiting.")
        return

    # --- 3. Train HMM ---
    print("\n--- Step 3: Training HMM ---")
    try:
        HmmTrader.Train(ProcessedData=ProcessedFeatures)
    except Exception as E:
        print(f"Error during HMM training: {E}. Exiting.")
        return

    if not HmmTrader.IsTrained:
        print("HMM model was not trained successfully. Exiting.")
        return

    print("HMM model trained successfully.")

    # --- 4. Predict Hidden States ---
    print("\n--- Step 4: Predicting Hidden States ---")
    try:
        PredictedStates = HmmTrader.PredictStates(Data=ProcessedFeatures)
        print(f"Predicted {len(PredictedStates)} hidden states.")
    except Exception as E:
        print(f"Error during state prediction: {E}. Exiting.")
        return

    # Align PredictedStates with the dates from HistoricalPriceData
    # ProcessedFeatures (and thus PredictedStates) are shorter due to initial NaNs from feature calculation
    if len(PredictedStates) == 0:
        print("No states were predicted. Cannot proceed with plotting. Exiting.")
        return

    # The data used for plotting should be the tail of HistoricalPriceData,
    # corresponding to the data points for which features could be computed.
    ValidPriceDataForPlotting = HistoricalPriceData.iloc[-len(PredictedStates):]

    if len(ValidPriceDataForPlotting) != len(PredictedStates):
        print(f"Critical Error: Length mismatch after alignment. Plotting will be incorrect. ValidPriceData: {len(ValidPriceDataForPlotting)}, States: {len(PredictedStates)}")
        # This case should ideally not happen if alignment logic is correct relative to PreprocessData
        return


    # --- 5. Generate Trading Signals (Commented out for now) ---
    # print("\n--- Step 5: Generating Trading Signals ---")
    # TradingSignals = HmmTrader.GenerateTradingSignals(PredictedStates=PredictedStates)
    # print(f"Generated {len(TradingSignals)} trading signals.")
    # print("Note: Signals are 1 (Buy), -1 (Sell), 0 (Hold/Neutral).")
    # print("Example Signals (first 10):")
    # if not TradingSignals.empty:
    #     print(TradingSignals.head(10))
    # else:
    #     print("No trading signals generated (possibly due to empty states).")


    # --- 6. Visualization ---
    print("\n--- Step 6: Plotting Results ---")
    if not ValidPriceDataForPlotting.empty:
        Figure, Axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # Plot 1: Close Prices and Hidden States
        Axes[0].plot(ValidPriceDataForPlotting.index, ValidPriceDataForPlotting['Close'], label="Close Price", color="blue", alpha=0.7)
        ColorMap = plt.get_cmap("jet", NumberOfHiddenStates)
        for I_State in range(NumberOfHiddenStates):
            StateMask = PredictedStates == I_State
            Axes[0].scatter(ValidPriceDataForPlotting.index[StateMask], ValidPriceDataForPlotting['Close'][StateMask],
                            color=ColorMap(I_State), label=f"State {I_State}", s=20, zorder=3) # zorder to plot points on top
        Axes[0].set_title(f"{TickerSymbol} Close Price and HMM Hidden States")
        Axes[0].set_ylabel("Price")
        Axes[0].legend()
        Axes[0].grid(True)

        # Plot 2: Hidden State Sequence
        Axes[1].plot(ValidPriceDataForPlotting.index, PredictedStates, label="Hidden State Sequence", color="red", marker='.', linestyle='none', markersize=5)
        Axes[1].set_title(f"{TickerSymbol} HMM Hidden State Sequence")
        Axes[1].set_ylabel("Hidden State")
        Axes[1].set_yticks(range(NumberOfHiddenStates)) # Ensure y-ticks match discrete states
        Axes[1].legend()
        Axes[1].grid(True)

        plt.xlabel("Date")
        plt.tight_layout()
        PlotFileName = f"{TickerSymbol}_Hmm_Trading_Analysis.png"
        plt.savefig(PlotFileName)
        print(f"Plot saved to {PlotFileName}")
        # plt.show() # Typically commented out for automated scripts
    else:
        print("Not enough data to plot or no predicted states available after alignment.")

    print("\n--- HMM Trading Demonstration Finished ---")

if __name__ == "__main__":
    Main()
