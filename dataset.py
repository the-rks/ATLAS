import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_headlines() -> pd.DataFrame:
    headlines_path = Path("Data/analyst_ratings_processed.csv")
    if not headlines_path.exists():
        print(
            "Please first download the analyst_ratings_processed.csv from:\n"
            "https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
        )
        webbrowser.open(
            "https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests"
        )
        raise FileNotFoundError()

    headlines = pd.read_csv(headlines_path)
    headlines = headlines.drop(headlines.columns[0], axis=1)
    return headlines


def load_stockprices(stock_picks: list[str]) -> pd.DataFrame:
    stock_prices: dict[str, pd.DataFrame] = {}

    for stock in stock_picks:
        stock_prices[stock] = pd.read_csv(f"Data/stocks/{stock}.csv")
        stock_prices[stock]["Date"] = pd.to_datetime(stock_prices[stock]["Date"])

    # merge into one df
    stockdata = None
    for stock, prices in stock_prices.items():
        if stockdata is None:
            stockdata = prices[["Date", "Close"]].rename(columns={"Close": stock})
        else:
            stockdata = stockdata.merge(
                prices[["Date", "Close"]], on="Date", how="outer"
            ).rename(columns={"Close": stock})
    stockdata = stockdata.sort_values("Date").reset_index(drop=True)

    return stockdata


def merge_daily(daily_headlines: pd.DataFrame, stockdata: pd.DataFrame) -> pd.DataFrame:
    """Combines (daily_headlines[date, stock] -> title) and
    (stockdata[Date, <stock_name>] -> price) to generate
    (merged[date, stock] -> headlines, price, delta), where headlines is the headlines
    of the stock from the date, price is the price of the stock the following date, and
    delta is the price change of the stock between the date and the following date.
    """
    # do not modify original
    daily_headlines = daily_headlines.copy()
    stockdata = stockdata.copy()

    daily_headlines["date"] = pd.to_datetime(daily_headlines["date"])
    stockdata["date"] = pd.to_datetime(stockdata["Date"])
    stockdata = stockdata.melt(id_vars=["date"], var_name="stock", value_name="price")

    stockdata_shifted = stockdata.copy()
    stockdata_shifted["date"] -= pd.Timedelta(days=1)

    merged = daily_headlines.merge(
        stockdata.rename(columns={"price": "prev_price"}),
        on=["date", "stock"],
        how="left",
    )
    merged = merged.merge(stockdata_shifted, on=["date", "stock"], how="left")

    merged["delta"] = merged["price"] - merged["prev_price"]
    merged = merged.rename(columns={"title": "headlines"})
    merged = merged.dropna(subset=["price", "delta"])
    merged = merged[["date", "stock", "headlines", "price", "delta"]].reset_index()
    return merged


def merge_weekly(
    weekly_headlines: pd.DataFrame, stockdata: pd.DataFrame
) -> pd.DataFrame:
    """Combines (weekly_headlines[week, stock] -> title) and
    (stockdata[Date, <stock_name>] -> price) to generate
    (merged[date, stock] -> headlines, price, delta), where headlines is the headlines
    of the stock from the date, price is the price of the stock the following week, and
    delta is the price change of the stock between the date and the following week.
    """
    # do not modify original
    weekly_headlines = weekly_headlines.copy()
    stockdata = stockdata.copy()

    weekly_headlines["date"] = pd.to_datetime(weekly_headlines["week"])
    stockdata["date"] = pd.to_datetime(stockdata["Date"])
    stockdata = stockdata.melt(id_vars=["date"], var_name="stock", value_name="price")

    stockdata_shifted = stockdata.copy()
    stockdata_shifted["date"] -= pd.Timedelta(days=7)

    merged = weekly_headlines.merge(
        stockdata.rename(columns={"price": "prev_price"}),
        on=["date", "stock"],
        how="left",
    )
    merged = merged.merge(stockdata_shifted, on=["date", "stock"], how="left")

    merged["delta"] = merged["price"] - merged["prev_price"]
    merged = merged.rename(columns={"title": "headlines"})
    merged = merged.dropna(subset=["price", "delta"])
    merged = merged[["date", "stock", "headlines", "price", "delta"]].reset_index()
    return merged


def get_document_scores(headlines: pd.DataFrame, max_tickers: int) -> pd.DataFrame:
    # train tf-idf and count vectorizer
    documents = headlines["title"].to_list()
    vectorizer_arg = {
        "stop_words": "english",
        "strip_accents": "ascii",
        "lowercase": True,
        "ngram_range": (1, 1),
        "max_df": 0.9,
        "min_df": 1,
    }
    tfidf_vec = TfidfVectorizer(**vectorizer_arg, norm=None)

    # get average tfidf score of words in documents
    tfidf = tfidf_vec.fit_transform(documents)
    document_scores = np.array(tfidf.sum(axis=1)).flatten()
    counts = tfidf.getnnz(axis=1)

    document_scores = np.divide(
        document_scores, counts, out=np.zeros_like(document_scores), where=(counts != 0)
    )  # ignore documents with score = 0 as we will get divide by 0

    # set score to 0 if there are n or more stock tickers consecutively
    reg = rf"(?:\b[A-Z]{{1,5}}\b[,]?\s*){{{max_tickers},}}\b[A-Z]{{1,5}}\b"
    document_scores[headlines["title"].str.contains(reg, regex=True)] = 0

    return document_scores
