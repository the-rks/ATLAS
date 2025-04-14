import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math


def load_articles(company_picks: dict) -> pd.DataFrame:
    articles_path = Path("Data/All_external.csv")
    if not articles_path.exists():
        print(
            "Please first download the All_external.csv from:\n"
            "https://huggingface.co/datasets/Zihan1004/FNSPID/blob/main/Stock_news/All_external.csv"
        )
        webbrowser.open(
            "https://huggingface.co/datasets/Zihan1004/FNSPID/blob/main/Stock_news/All_external.csv"
        )
        raise FileNotFoundError()

    all_articles = pd.read_csv(articles_path)

    # Drop all rows that do no have the article column filled out
    all_articles.dropna(inplace=True, axis=0, how='all', subset=['Article'])
    all_articles.reset_index(inplace=True)

    # Drop all articles that are in Russian
    last_russain_index = 800969
    drop_idx = [i for i in range(last_russain_index + 1)]
    all_articles.drop(inplace=True, index=drop_idx)
    all_articles.reset_index(inplace=True)
    condensed_articles = all_articles.drop(columns=['index', 'level_0', 'Stock_symbol',
                                            'Url', 'Publisher', 'Author','Lsa_summary', 'Luhn_summary',
                                            'Textrank_summary', 'Lexrank_summary']) # Unnecessary columns to drop

    # Set a column that contains the stock symbol for each article
    # Drop articles that are not relevant (i.e. none of the company names in their title)
    condensed_articles.insert(1, 'Stock', "Not set")
    to_drop = []
    for idx, row in condensed_articles.iterrows():
        title = row['Article_title']
        in_picks = False
        for comp_name, stock_name in company_picks.items():
            if comp_name in title:
                condensed_articles.loc[idx, 'Stock'] = stock_name
                in_picks = True
                break
        if not in_picks:
            to_drop.append(idx)
    final_articles = condensed_articles.drop(index=to_drop)
    final_articles = final_articles.reset_index().drop(columns='index')

    return final_articles

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

def add_deltas(articles: pd.DataFrame, stockdata: pd.DataFrame) -> pd.DataFrame:
    # Make copies of datasets
    c_articles = articles.copy()
    c_stockdata = stockdata.copy()

    # Insert columns for deltas (difference in stock after a week/month from publication date of article)
    articles.insert(4, 'delta_week', math.nan)
    articles.insert(5, 'delta_month', math.nan)

    # Convert date column of article dataset to appropriate format
    c_articles['Date'] = c_articles['Date'].apply(lambda x: x[:10])
    c_articles['Date'] = pd.to_datetime(c_articles["Date"])

    for idx, row in c_articles.iterrows():
        date = row['Date']
        stock = row['Stock']

        # Adjust date if needed (no stock prices on weekends)
        if (c_stockdata['Date'] == date).sum() == 0:
            date = date + pd.Timedelta(days=1)
            if (c_stockdata['Date'] == date).sum() == 0:
                date = date + pd.Timedelta(days=1)
                if (c_stockdata['Date'] == date).sum() == 0:
                    date = date + pd.Timedelta(days=1)

        # Set the dates for a week (adjust if needed in case of holiday date)
        week_date = date + pd.Timedelta(days=7)
        if (c_stockdata['Date'] == week_date).sum() == 0:
            week_date = week_date + pd.Timedelta(days=1)
            if (c_stockdata['Date'] == week_date).sum() == 0:
                week_date = week_date + pd.Timedelta(days=1)
                if (c_stockdata['Date'] == week_date).sum() == 0:
                    week_date = week_date + pd.Timedelta(days=1)

        # Set the dates for a month (adjust if needed in case of holiday date)
        month_date = date + pd.Timedelta(days=28)
        if (c_stockdata['Date'] == month_date).sum() == 0:
            month_date = month_date + pd.Timedelta(days=1)
            if (c_stockdata['Date'] == month_date).sum() == 0:
                month_date = month_date + pd.Timedelta(days=1)
                if (c_stockdata['Date'] == month_date).sum() == 0:
                    month_date = month_date + pd.Timedelta(days=1)

        # Add price deltas to the original dataset
        curr_price = c_stockdata.loc[c_stockdata['Date'] == date].reset_index()[stock][0]
        week_price = c_stockdata.loc[c_stockdata['Date'] == week_date].reset_index()[stock][0]
        month_price = c_stockdata.loc[c_stockdata['Date'] == month_date].reset_index()[stock][0]
        delta_week = week_price - curr_price
        delta_month = month_price - curr_price
        articles.loc[idx, 'delta_week'] = delta_week
        articles.loc[idx, 'delta_month'] = delta_month

    return articles
