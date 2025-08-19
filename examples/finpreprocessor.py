# Financial Data Preprocessing

import os
import sys

REPO_ROOT = "/Users/ingridperez/Documents/GitHub Repositiories/xdl-portfolio"
sys.path.append(REPO_ROOT)

from config import config, config_indicators
from preprocessor.findata_downloader import FinancialDataDownloader
from preprocessor.findata_preprocessor import FinancialDataPreprocessor
from visualiser.findata_visualiser import FinancialDataVisualiser

data_dir = f"{REPO_ROOT}/{config.DATA_DIR}/{config.DATASET_NAME}"
plot_dir = (
    f"{REPO_ROOT}/{config.PLOT_DIR}/{config.TICKERS_NAME}/{config.DATASET_NAME}"
)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

findownloader = FinancialDataDownloader(
    start_date=config.START_DATE,
    end_date=config.END_DATE,
)
data = findownloader.load_data(
    directory=data_dir,
    filename=config.TICKERS_NAME,
)

finpreprocessor = FinancialDataPreprocessor(
    start_date=config.START_DATE,
    end_date=config.END_DATE,
)
data = finpreprocessor.preprocess(
    data=data,
    exchange=config.EXCHANGE,
    use_tech_indicators=config.USE_TECHNICAL_INDICATORS,
    tech_indicators=list(config_indicators.TECHNICAL_INDICATORS.keys()),
    use_macro_indicators=config.USE_MACROECONOMIC_INDICATORS,
    macro_indicators=(
        list(config.MACROECONOMIC_INDICATORS.keys())
        if config.USE_MACROECONOMIC_INDICATORS
        else []
    ),
    use_covariance=config.USE_COVARIANCE_FEATURES,
)

finvisualiser = FinancialDataVisualiser(directory=plot_dir)
finvisualiser.plot_close_prices(
    data=data,
    filename="processed_close_prices",
)

if config.USE_TECHNICAL_INDICATORS:
    finvisualiser.plot_technical_indicators(
        data=data,
        indicators=config_indicators.TECHNICAL_INDICATORS,
    )

if config.USE_MACROECONOMIC_INDICATORS:
    finvisualiser.plot_macroeconomic_indicators(
        data=data,
        indicators=config.MACROECONOMIC_INDICATORS,
    )

train_data, test_data = finpreprocessor.split_train_test(
    data=data,
    test_start_date=config.TEST_START_DATE,
)

finvisualiser.plot_train_test_close_prices(
    train_data=train_data,
    test_data=test_data,
)

finpreprocessor.save_train_test_data(
    train_data=train_data,
    test_data=test_data,
    directory=data_dir,
    filename=config.TICKERS_NAME,
)
