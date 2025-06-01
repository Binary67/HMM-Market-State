import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from HmmTrading.DataDownloader import YahooFinanceDownloader
from HmmTrading.HmmModel import TradingHmm

# Set page config for a wider layout
st.set_page_config(layout="wide")

# App title
st.title("HMM Trading Analysis")

# Sidebar for inputs
st.sidebar.header("User Input Parameters")

DefaultTicker = "AAPL"
TickerSymbol = st.sidebar.text_input("Ticker Symbol", DefaultTicker)

DefaultStartDate = datetime.date(2020, 1, 1)
StartDate = st.sidebar.date_input("Start Date", DefaultStartDate)

DefaultEndDate = datetime.date.today()
EndDate = st.sidebar.date_input("End Date", DefaultEndDate)

DefaultNumStates = 3
NumberOfHiddenStates = st.sidebar.number_input("Number of Hidden States", min_value=2, max_value=10, value=DefaultNumStates, step=1)

RunButton = st.sidebar.button("Run Analysis")

if RunButton:
    # Convert dates to string format
    StartDateStr = StartDate.strftime("%Y-%m-%d")
    EndDateStr = EndDate.strftime("%Y-%m-%d")

    st.info("Starting analysis...")

    # Download data
    DataDownloaderInstance = YahooFinanceDownloader(TickerSymbol=TickerSymbol)
    HistoricalPriceData = DataDownloaderInstance.DownloadData(StartDate=StartDateStr, EndDate=EndDateStr)

    if HistoricalPriceData.empty:
        st.error(f"No data downloaded for {TickerSymbol}. Please check the ticker symbol and date range.")
        st.stop()
    else:
        st.success(f"Successfully downloaded data for {TickerSymbol}.")

    # Instantiate HMM Trader
    HmmTrader = TradingHmm(NumberOfHiddenStates=NumberOfHiddenStates)

    # Preprocess data
    try:
        ProcessedFeatures = HmmTrader.PreprocessData(PriceData=HistoricalPriceData)
        st.success("Data preprocessed successfully.")
        if ProcessedFeatures.shape[0] == 0:
            st.warning("No features available after preprocessing. The date range might be too short or data unsuitable. Cannot proceed.")
            st.stop()
    except ValueError as VE:
        st.error(f"Error during data preprocessing: {VE}")
        st.stop()

    # Train HMM
    try:
        HmmTrader.Train(ProcessedData=ProcessedFeatures)
        st.success("HMM model trained successfully.")
        if not HmmTrader.IsTrained: # Should not happen if no exception, but good practice
            st.error("HMM model was not trained successfully. Exiting.")
            st.stop()
    except Exception as E:
        st.error(f"Error during HMM training: {E}")
        st.stop()

    # Predict States
    try:
        PredictedStates = HmmTrader.PredictStates(Data=ProcessedFeatures)
        st.success("Hidden states predicted successfully.")
        if len(PredictedStates) == 0:
            st.warning("No states were predicted. Cannot proceed with plotting.")
            st.stop()
    except Exception as E:
        st.error(f"Error during state prediction: {E}")
        st.stop()

    # Store results in session state
    st.session_state.HistoricalPriceData = HistoricalPriceData
    st.session_state.PredictedStates = PredictedStates
    st.session_state.HmmTrader = HmmTrader
    st.session_state.ProcessedFeatures = ProcessedFeatures
    # Save the TickerSymbol to session_state when the button is pressed, to be used in the plot title
    st.session_state.TickerSymbol = TickerSymbol
    st.session_state.RunButton = True # Flag that run button was pressed

# Initialize RunButton state if not already present, for initial load behavior
if 'RunButton' not in st.session_state:
    st.session_state.RunButton = False

# --- Display Results ---
if 'PredictedStates' in st.session_state and \
   'HistoricalPriceData' in st.session_state and \
   'HmmTrader' in st.session_state and \
   'ProcessedFeatures' in st.session_state:

    PredictedStates = st.session_state.PredictedStates
    HistoricalPriceData = st.session_state.HistoricalPriceData
    HmmTrader = st.session_state.HmmTrader # For NumberOfHiddenStates for colormap
    ProcessedFeatures = st.session_state.ProcessedFeatures # To align data for plotting

    # Align PredictedStates with the dates from HistoricalPriceData
    if len(PredictedStates) > 0 and len(HistoricalPriceData) >= len(ProcessedFeatures):
        # The length of ProcessedFeatures should be the same as PredictedStates
        # HistoricalPriceData needs to be trimmed from the start to align with ProcessedFeatures/PredictedStates
        # The most robust way is to align based on the length of PredictedStates.
        ValidPriceDataForPlotting = HistoricalPriceData.iloc[-len(PredictedStates):]

        if len(ValidPriceDataForPlotting) == len(PredictedStates):
            st.subheader(f"Analysis Results for {st.session_state.get('TickerSymbol', TickerSymbol)}")

            Figure, Axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
            plt.style.use('seaborn-v0_8-darkgrid')

            # Plot 1: Close Prices and Hidden States
            Axes[0].plot(ValidPriceDataForPlotting.index, ValidPriceDataForPlotting['Close'], label="Close Price", color="blue", alpha=0.7)

            NumberOfHiddenStatesFromModel = HmmTrader.NumberOfHiddenStates
            ColorMap = plt.get_cmap("jet", NumberOfHiddenStatesFromModel)

            for I_State in range(NumberOfHiddenStatesFromModel):
                StateMask = (PredictedStates == I_State)
                Axes[0].scatter(ValidPriceDataForPlotting.index[StateMask], ValidPriceDataForPlotting['Close'][StateMask],
                                color=ColorMap(I_State), label=f"State {I_State}", s=25, zorder=3, alpha=0.8)
            Axes[0].set_title(f"Close Price and HMM Hidden States")
            Axes[0].set_ylabel("Price")
            Axes[0].legend(loc="upper left")
            Axes[0].grid(True, which='both', linestyle='--', linewidth=0.5)

            # Plot 2: Hidden State Sequence
            Axes[1].plot(ValidPriceDataForPlotting.index, PredictedStates, label="Hidden State Sequence", color="red", marker='o', linestyle='-', markersize=4, alpha=0.7)
            Axes[1].set_title(f"HMM Hidden State Sequence")
            Axes[1].set_ylabel("Hidden State")
            Axes[1].set_yticks(range(NumberOfHiddenStatesFromModel))
            Axes[1].set_yticklabels([str(i) for i in range(NumberOfHiddenStatesFromModel)])
            Axes[1].legend(loc="upper left")
            Axes[1].grid(True, which='both', linestyle='--', linewidth=0.5)

            plt.xlabel("Date")
            plt.tight_layout()
            st.pyplot(Figure)

            with st.expander("View HMM Model Parameters"):
                st.write("Transition Matrix:")
                st.dataframe(pd.DataFrame(HmmTrader.Model.transmat_))
                st.write("Means:")
                st.dataframe(pd.DataFrame(HmmTrader.Model.means_))
                # Robustly display covariances
                Covars = HmmTrader.Model.covars_
                if Covars.ndim == 2: # Typically diagonal covariances, one row per state
                    st.write("Covariances (Diagonal Elements per State):")
                    StCovarsDf = pd.DataFrame(Covars)
                    StCovarsDf.columns = [f'Feature_{j}' for j in range(Covars.shape[1])]
                    StCovarsDf.index = [f'State_{i}' for i in range(Covars.shape[0])]
                    st.dataframe(StCovarsDf)
                elif Covars.ndim == 3: # Full covariance matrices, one matrix per state
                    st.write("Covariance Matrices per State:")
                    for I_State in range(Covars.shape[0]):
                        st.write(f"State {I_State} Covariance Matrix:")
                        StStateCovDf = pd.DataFrame(Covars[I_State])
                        # Assuming features correspond to columns and rows in the covariance matrix
                        StStateCovDf.columns = [f'Feature_{j}' for j in range(Covars.shape[2])]
                        StStateCovDf.index = [f'Feature_{i}' for i in range(Covars.shape[1])]
                        st.dataframe(StStateCovDf)
                else:
                    st.write("Covariances structure not recognized (expected 2D or 3D array).")
                    st.write(f"Shape: {Covars.shape}")
                    st.write("Raw Covariances:")
                    st.write(Covars) # Display raw covariances if structure is not as expected
        else:
            st.warning("Could not align data for plotting. Length mismatch between price data and predicted states.")
            st.write(f"Length of ValidPriceDataForPlotting: {len(ValidPriceDataForPlotting)}")
            st.write(f"Length of PredictedStates: {len(PredictedStates)}")
            st.write(f"Original Historical Price Data Length: {len(HistoricalPriceData)}")
            st.write(f"Processed Features Length: {len(ProcessedFeatures)}")

    elif st.session_state.RunButton: # Check if button was pressed to avoid showing on first load if no data
        st.warning("Not enough data to plot or no predicted states available after alignment.")
