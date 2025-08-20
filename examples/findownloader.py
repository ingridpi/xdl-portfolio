# Financial Data Downloader example

import os
import sys

REPO_ROOT = "/Users/ingridperez/Documents/GitHub Repositories/xdl-portfolio"
sys.path.append(REPO_ROOT)

from config import config
from preprocessor.findata_downloader import FinancialDataDownloader
from visualiser.findata_visualiser import FinancialDataVisualiser

data_dir = f"{REPO_ROOT}/{config.DATA_DIR}/{config.DATASET_NAME}"
plot_dir = (
    f"{REPO_ROOT}/{config.PLOT_DIR}/{config.TICKERS_NAME}/{config.DATASET_NAME}"
)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

findownloader = FinancialDataDownloader(
    start_date=config.START_DATE,
    end_date=config.END_DATE,
)

data = findownloader.download_data(tickers=config.TICKERS)

findownloader.save_data(
    directory=data_dir,
    filename=config.TICKERS_NAME,
)

finvisualiser = FinancialDataVisualiser(directory=plot_dir)
finvisualiser.plot_close_prices(data=data)
