import json
from datetime import datetime, timedelta
from pathlib import Path
from textwrap import indent
from typing import TextIO, TypedDict

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from googlesearch import search
from tqdm import tqdm


def setup_env():
    load_dotenv()
    # check if certain envs exist and prompt user if not, saving it to .env


class SearchArg(TypedDict):
    name: str
    from_date: datetime.date
    to_date: datetime.date
    date_range: int  # will grab news on dates [from_date:to_date:date_range]


AREAS = [
    {"name": "title"},
    {"name": "div", "class_": "ArticleBody-articleBody"},
    {"name": "div", "class_": "atoms-wrapper"},
    {"name": "div", "class_": "article-body"},
]


def process_link(url: str) -> str | None:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=1.5)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # extract content
        content = (
            " ".join(
                " ".join(
                    elem.get_text(separator=" ", strip=True)
                    for elem in soup.find_all(**area)
                )
                for area in AREAS
                if soup.find(**area)
            )
            .strip()
            .lower()
        )

        return content
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None


def gather_stocknews(
    name: str, from_date: datetime.date, to_date: datetime.date, outfile: TextIO
):
    filters = f"after:{from_date} before:{to_date} "
    queries = [
        f'site:finance.yahoo.com {name} + "in this article"',
        # f'site:cnbc.com {name} + "in this article"',  # rate limited
        # f'site:reuters.com {name} + "Suggested Topics"',  # paywall
        f"site:fool.com {name}",
    ]

    stock_texts = []
    for query in queries:
        result = search(f"{query} {filters}", num_results=2)

        for link in result:
            if not link:
                continue
            site_data = process_link(link)
            if site_data is not None:
                stock_texts.append(site_data)

    writedata = json.dumps(
        {"name": name, "date": str(from_date), "text": stock_texts}, indent=2
    )
    writedata = indent(writedata, "  ") + ",\n"
    outfile.write(writedata)


def gather_news(search_args: list[SearchArg], output: Path = Path("output.csv")):
    print(f"Starting search. Output in {output}")
    with open(output, "w", newline="") as outfile:
        outfile.write("[\n")

        for arg in search_args:
            name = arg["name"]
            from_date = arg["from_date"]
            to_date = arg["to_date"]
            date_range = arg["date_range"]
            date_incr = timedelta(days=date_range)

            print(f"Gathering news for {name} from {from_date} to {to_date}")
            n_iter = (to_date - from_date) // date_incr
            for _ in tqdm(range(n_iter), unit="interval"):
                gather_stocknews(name, from_date, from_date + date_incr, outfile)
                from_date += date_incr

        outfile.write("]\n")


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

    setup_env()
    gather_news(search_args, Path("Data/stocknews.csv"))
