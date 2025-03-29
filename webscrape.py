import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import TypedDict

from googlesearch import search
from tqdm import tqdm


class SearchArg(TypedDict):
    name: str
    from_date: datetime.date  # inclusive
    to_date: datetime.date  # inclusive
    date_range: int  # will grab news on dates [from_date:to_date:date_range]


def get_news_links(company, start_date, end_date, max_results=10):
    # Restrict search to free sources
    query = (
        f"{company} stock news after:{start_date} before:{end_date} "
        f"site:finance.yahoo.com OR site:investing.com OR site:seekingalpha.com"
    )
    news_links = []

    for url in search(query, num_results=max_results):
        if any(
            domain in url
            for domain in ["yahoo.com", "investing.com", "seekingalpha.com"]
        ):
            news_links.append(url)

    return news_links


def process_link(link: str) -> str | None:
    pass


def gather_stocknews(
    name: str, from_date: datetime.date, to_date: datetime.date, writer: csv.writer
):
    filters = f"after:{from_date} before:{to_date} "
    queries = [
        f"site:finance.yahoo.com {name} stock article",
        f'site:cnbc.com {name} + "published"',
        f'site:reuters.com {name} + "Suggested Topics"',
        f"site:fool.com {name}",
    ]

    stock_texts = []
    for query in queries:
        result = search(f"{query} {filters}", num_results=2)

        for link in result:
            site_data = process_link(link)
            if site_data is not None:
                stock_texts.append(site_data)

    text_lists = json.dumps(stock_texts)
    writer.writerows([str(from_date), name, text_lists])


def gather_news(search_args: list[SearchArg], output: Path = Path("output.csv")):
    print(f"Starting search. Output in {output}")
    with open(output, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["date", "name", "text_list"])

        for arg in search_args:
            name = arg["name"]
            from_date = arg["from_date"]
            to_date = arg["to_date"]
            date_range = arg["date_range"]
            date_incr = timedelta(days=date_range)

            print(f"Gathering news for {name} from {from_date} to {to_date}")
            n_iter = (to_date - from_date) // date_incr
            for _ in tqdm(range(n_iter), unit="interval"):
                gather_stocknews(name, from_date, from_date + date_incr, writer)
                from_date += date_incr


def date(datestr: str) -> datetime.date:
    """Convert date of form MM/DD/YY to a datetime date object."""
    return datetime.strptime(datestr, "%m/%d/%Y").date()


if __name__ == "__main__":
    date_range = 7  # get weekly stock news
    search_args: list[SearchArg] = [
        {
            "name": "Apple",
            "from_date": date("3/27/2020"),
            "to_date": date("3/21/2025"),
            "date_range": date_range,
        },
        {
            "name": "Boeing",
            "from_date": date("3/27/2020"),
            "to_date": date("3/21/2025"),
            "date_range": date_range,
        },
        {
            "name": "Nvidia",
            "from_date": date("11/12/2021"),
            "to_date": date("3/21/2025"),
            "date_range": date_range,
        },
        {
            "name": "Palantir",
            "from_date": date("10/2/2020"),
            "to_date": date("3/21/2025"),
            "date_range": date_range,
        },
        {
            "name": "Pfizer",
            "from_date": date("3/27/2020"),
            "to_date": date("3/21/2025"),
            "date_range": date_range,
        },
        {
            "name": "Tesla",
            "from_date": date("3/27/2020"),
            "to_date": date("3/21/2025"),
            "date_range": date_range,
        },
        {
            "name": "Netflix",
            "from_date": date("3/27/2020"),
            "to_date": date("3/21/2025"),
            "date_range": date_range,
        },
        {
            "name": "Meta",
            "from_date": date("3/27/2020"),
            "to_date": date("3/21/2025"),
            "date_range": date_range,
        },
        {
            "name": "Intel",
            "from_date": date("3/27/2020"),
            "to_date": date("3/21/2025"),
            "date_range": date_range,
        },
        {
            "name": "Lockheed Martin",
            "from_date": date("3/27/2020"),
            "to_date": date("3/21/2025"),
            "date_range": date_range,
        },
        {
            "name": "Rivian",
            "from_date": date("11/12/2021"),
            "to_date": date("3/21/2025"),
            "date_range": date_range,
        },
    ]

    gather_news(search_args, Path("Data/stocknews.csv"))
