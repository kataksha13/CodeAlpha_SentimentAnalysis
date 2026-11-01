# CodeAlpha_SentimentAnalysis

**CodeAlpha Data Analytics Internship, Task 4: Sentiment Analysis**

An NLP project that classifies text as **positive, negative or neutral**. It scrapes quotes from
[Quotes to Scrape](https://quotes.toscrape.com) (a sandbox site built for practicing web scraping)
and scores each one with **VADER**, a lexicon-based sentiment model.

The analysis script also works on any other CSV with a text column, for example Amazon product reviews.

## How it works

1. `scrape_quotes(1).py` collects quotes, authors and tags with requests and BeautifulSoup.
2. `sentiment.py` cleans the text (links, HTML tags, extra spaces).
3. VADER gives each text a **compound score** from -1 (very negative) to +1 (very positive).
4. The score is turned into a label using the standard thresholds:
   - Positive: score >= 0.05
   - Negative: score <= -0.05
   - Neutral: in between
5. Results are summarised with charts and saved to a CSV and a text report.

## What the analysis covers

- Overall share of positive, neutral and negative texts
- Distribution of sentiment scores
- Most positive and most negative texts
- Average sentiment by author and by tag
- Most common words in positive vs negative texts
- Whether text length relates to sentiment

## Key findings
- __% of the quotes are positive, __% neutral and __% negative.
- The average compound score is __, so the collection leans (positive / negative / neutral).
- Authors / tags with the most positive sentiment: __. Most negative: __.
- Common words in positive texts: __. In negative texts: __.

## Limitations

- VADER is lexicon-based: it counts known emotional words and does not truly understand context.
- Sarcasm, irony and deeper meaning (common in quotes) can be scored wrongly.
- Quotes are short and abstract, so many score as neutral. Product reviews usually give clearer results.

## How to run

```bash
pip install -r requirements.txt
python scrape_quotes.py       # creates data/quotes.csv
python sentiment.py           # creates charts/, sentiment_report.txt and data/sentiment_results.csv
```

To analyse your own file (for example reviews downloaded from Kaggle):

```bash
python sentiment.py --file data/reviews.csv --text-column review_text
```

## Project structure

```
CodeAlpha_SentimentAnalysis/
├── data/
│   ├── quotes.csv
│   └── sentiment_results.csv
├── charts/
├── scrape_quotes.py
├── sentiment.py
├── sentiment_report.txt
├── requirements.txt
└── README.md
```

## Tools

Python, requests, BeautifulSoup4, pandas, VADER (vaderSentiment), matplotlib, seaborn
