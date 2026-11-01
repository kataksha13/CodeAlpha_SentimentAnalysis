"""
CodeAlpha Data Analytics Internship - Task 4: Sentiment Analysis (step 2)
Classifies text as positive, negative or neutral using VADER, a lexicon-based NLP model.

Usage:
    python sentiment.py                                   # analyses data/quotes.csv
    python sentiment.py --file data/reviews.csv --text-column review_text
                                                          # any CSV with a text column (e.g. Amazon reviews)

Outputs:
    data/sentiment_results.csv   every row with its score and label
    charts/*.png                 visuals
    sentiment_report.txt         summary of findings
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CHART_DIR = Path("charts")
COLORS = {"Positive": "#3C9D6B", "Neutral": "#9AA5B1", "Negative": "#C0504D"}
REPORT = []

sns.set_theme(style="whitegrid")
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Common words that carry no sentiment, ignored when counting top words
STOPWORDS = set(
    """a about above after again all also am an and any are as at be because been before being below between both
    but by can could did do does doing down during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on
    once only or other our ours ourselves out over own same she should so some such than that the their theirs them
    themselves then there these they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours yourself yourselves one like get got us even much many may
    must never always every something nothing anything everything""".split()
)


def say(text=""):
    print(text)
    REPORT.append(str(text))


def save(fig, name):
    path = CHART_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    say(f"[chart saved: {path}]")


# ---------------------------------------------------------------------------
# 1. Clean and score
# ---------------------------------------------------------------------------
def clean_text(text):
    """Light cleaning. VADER works best on raw-ish text, so we keep punctuation and capitals."""
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove links
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
    return re.sub(r"\s+", " ", text).strip()


def label_from_score(score):
    """Standard VADER thresholds."""
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def score_texts(df, text_col):
    analyzer = SentimentIntensityAnalyzer()
    df = df.copy()
    df["clean_text"] = df[text_col].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    scores = df["clean_text"].apply(analyzer.polarity_scores)
    df["compound"] = scores.apply(lambda s: s["compound"])
    df["pos"] = scores.apply(lambda s: s["pos"])
    df["neu"] = scores.apply(lambda s: s["neu"])
    df["neg"] = scores.apply(lambda s: s["neg"])
    df["sentiment"] = df["compound"].apply(label_from_score)
    return df


# ---------------------------------------------------------------------------
# 2. Analysis and charts
# ---------------------------------------------------------------------------
def overall(df):
    say("=" * 70)
    say("OVERALL SENTIMENT")
    say("=" * 70)
    counts = df["sentiment"].value_counts().reindex(["Positive", "Neutral", "Negative"], fill_value=0)
    pct = (counts / len(df) * 100).round(1)
    for label in counts.index:
        say(f"{label:<9} {counts[label]:>5}  ({pct[label]}%)")
    say(f"\nAverage compound score: {df['compound'].mean():.3f}  (-1 = very negative, +1 = very positive)")
    say(f"Median compound score : {df['compound'].median():.3f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(counts.index, counts.values, color=[COLORS[l] for l in counts.index], edgecolor="white")
    for bar, v, p in zip(bars, counts.values, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + counts.max() * 0.01, f"{v}\n({p}%)", ha="center", fontsize=10)
    ax.set_title("Sentiment Distribution", loc="left", fontweight="bold")
    ax.set_ylabel("Number of texts")
    ax.set_ylim(0, counts.max() * 1.18)
    save(fig, "01_sentiment_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["compound"], bins=30, color="#2E6F9E", edgecolor="white", ax=ax)
    ax.axvline(0.05, color="grey", linestyle="--", linewidth=1)
    ax.axvline(-0.05, color="grey", linestyle="--", linewidth=1)
    ax.set_title("Distribution of Sentiment Scores", loc="left", fontweight="bold")
    ax.set_xlabel("Compound score (dashed lines mark the neutral zone)")
    ax.set_ylabel("Number of texts")
    save(fig, "02_score_distribution.png")


def examples(df, text_col):
    say("\n" + "=" * 70)
    say("MOST POSITIVE AND MOST NEGATIVE TEXTS")
    say("=" * 70)
    say("Top 5 most positive:")
    for _, r in df.nlargest(5, "compound").iterrows():
        say(f"  [{r['compound']:+.2f}] {r['clean_text'][:140]}")
    say("\nTop 5 most negative:")
    for _, r in df.nsmallest(5, "compound").iterrows():
        say(f"  [{r['compound']:+.2f}] {r['clean_text'][:140]}")


def by_group(df, col, title, filename, min_count, top_n=10):
    """Average sentiment per group (author or tag)."""
    g = df.groupby(col)["compound"].agg(["count", "mean"]).query("count >= @min_count")
    if g.empty:
        say(f"\n(No {col} has at least {min_count} texts, skipping that chart.)")
        return
    g = g.sort_values("count", ascending=False).head(top_n).sort_values("mean")

    say(f"\n{title}")
    say(g.round(3).sort_values("mean", ascending=False).to_string())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [COLORS["Positive"] if m >= 0 else COLORS["Negative"] for m in g["mean"]]
    ax.barh(g.index, g["mean"], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Average compound score")
    ax.set_ylabel("")
    save(fig, filename)


def groups(df):
    say("\n" + "=" * 70)
    say("SENTIMENT BY GROUP")
    say("=" * 70)

    if "author" in df.columns:
        by_group(df, "author", "Average sentiment by author (authors with 3+ texts)",
                 "03_sentiment_by_author.png", min_count=3)

    if "tags" in df.columns:
        tag_df = df.assign(tag=df["tags"].fillna("").str.split(",")).explode("tag")
        tag_df = tag_df[tag_df["tag"].str.strip() != ""]
        by_group(tag_df, "tag", "Average sentiment by tag (tags with 5+ texts)",
                 "04_sentiment_by_tag.png", min_count=5)


def top_words(df):
    say("\n" + "=" * 70)
    say("MOST COMMON WORDS IN POSITIVE VS NEGATIVE TEXTS")
    say("=" * 70)

    def count_words(texts):
        words = re.findall(r"[a-z']+", " ".join(texts).lower())
        return Counter(w for w in words if w not in STOPWORDS and len(w) > 2)

    pos = count_words(df.loc[df["sentiment"] == "Positive", "clean_text"]).most_common(10)
    neg = count_words(df.loc[df["sentiment"] == "Negative", "clean_text"]).most_common(10)
    say("Positive: " + ", ".join(f"{w} ({c})" for w, c in pos))
    say("Negative: " + ", ".join(f"{w} ({c})" for w, c in neg))

    if not pos or not neg:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, label in zip(axes, (pos, neg), ("Positive", "Negative")):
        words, counts = zip(*data[::-1])
        ax.barh(words, counts, color=COLORS[label], edgecolor="white")
        ax.set_title(f"Top words in {label.lower()} texts", loc="left", fontweight="bold")
        ax.set_xlabel("Count")
    save(fig, "05_top_words.png")


def text_length(df):
    df["word_count"] = df["clean_text"].str.split().str.len()
    corr = df["word_count"].corr(df["compound"])
    say("\n" + "=" * 70)
    say("DOES LENGTH MATTER?")
    say("=" * 70)
    say(f"Correlation between text length (words) and sentiment score: {corr:.3f}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["word_count"], df["compound"], alpha=0.5, color="#7B5EA7", edgecolor="none")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title("Text Length vs Sentiment Score", loc="left", fontweight="bold")
    ax.set_xlabel("Words in text")
    ax.set_ylabel("Compound score")
    save(fig, "06_length_vs_sentiment.png")


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Sentiment analysis with VADER")
    parser.add_argument("--file", default="data/quotes.csv", help="CSV file to analyse")
    parser.add_argument("--text-column", default="quote", help="name of the column that holds the text")
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Could not find {path}. For the default run, first do: python scrape_quotes.py")
        return

    raw = pd.read_csv(path)
    if args.text_column not in raw.columns:
        print(f"Column '{args.text_column}' not found. Available columns: {list(raw.columns)}")
        print("Use --text-column to pick the right one.")
        return

    CHART_DIR.mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    df = score_texts(raw, args.text_column)
    say(f"Analysed {len(df)} texts from {path}\n")

    overall(df)
    examples(df, args.text_column)
    groups(df)
    top_words(df)
    text_length(df)

    df.drop(columns=["clean_text"]).to_csv("data/sentiment_results.csv", index=False, encoding="utf-8-sig")
    Path("sentiment_report.txt").write_text("\n".join(REPORT), encoding="utf-8")
    print("\nSaved: data/sentiment_results.csv, sentiment_report.txt and charts/")


if __name__ == "__main__":
    main()
