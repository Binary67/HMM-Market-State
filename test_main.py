from main import Main


def test_main_returns_dataframe_with_regime():
    Data = Main()
    assert "Regime" in Data.columns
    assert not Data.empty
