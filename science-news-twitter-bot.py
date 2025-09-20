"""
Science News Twitter Bot (REST OpenAI + Policy Safe)

Features:
- Pulls curated RSS feeds (AI, quantum, fusion, longevity, BMI)
- Allowlist of reputable domains + fringe keyword blocklist
- 7-day recency filter (configurable)
- Labels arXiv items as [Preprint]
- Uses OpenAI Chat Completions via REST (requests) to avoid SDK constructor issues in Actions
- SQLite dedupe store
- Dry-run mode for previews and real post mode using Tweepy (OAuth1)
"""
from __future__ import annotations
import os
import sys
import re
import time
import json
import textwrap
import argparse
import sqlite3
import random
import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Optional

import feedparser
import requests
from bs4 import BeautifulSoup
import tweepy
from dotenv import load_dotenv
import tldextract

# ---------------- Config / Defaults ----------------

FOCUS_HASHTAGS = [
    "#AI", "#QuantumComputing", "#Fusion", "#Longevity", "#Aging", "#Neuroscience",
    "#BrainMachineInterface", "#LifeExtension", "#Biotech", "#ScienceNews"
]

RSS_SOURCES: Dict[str, List[str]] = {
    "arxiv_ai": ["https://export.arxiv.org/rss/cs.AI", "https://export.arxiv.org/rss/cs.LG"],
    "deepmind": ["https://deepmind.google/atom.xml"],
    "openai": ["https://openai.com/blog/rss.xml"],
    "anthropic": ["https://www.anthropic.com/news/rss.xml"],
    "mit_csail": ["https://www.csail.mit.edu/rss.xml"],
    "arxiv_quantum": ["https://export.arxiv.org/rss/quant-ph"],
    "quantamag": ["https://api.quantamagazine.org/feed/"],
    "iter": ["https://www.iter.org/rss"],
    "pppl": ["https://www.pppl.gov/news.xml"],
    "nature_aging": ["https://www.nature.com/subjects/ageing/rss"],
    "science": ["https://www.science.org/rss/news_current.xml"],
    "nature_news": ["https://www.nature.com/nature/articles?type=news-and-comment.rss"],
    "nature_neuro": ["https://www.nature.com/subjects/neuroscience/rss"],
    "nih_news": ["https://www.nih.gov/news-events/news-releases/feed"],
    "ieee_spectrum": ["https://spectrum.ieee.org/feed"],
}

ALLOWED_DOMAINS = {
    "nature.com","science.org","cell.com","thelancet.com","nejm.org",
    "quantamagazine.org","spectrum.ieee.org","ieee.org","mit.edu","csail.mit.edu",
    "stanford.edu","nih.gov","pppl.gov","iter.org","openai.com",
    "deepmind.google","anthropic.com","arxiv.org"
}

BLOCK_KEYWORDS = {"free energy","overunity","telepathy","miracle cure","ancient aliens"}

MAX_TWEET_LEN = 280
DB_PATH = os.environ.get("SCI_BOT_DB", "science_bot.db")
USER_AGENT = "SciNewsBot/1.0 (https://github.com/)"

# OpenAI REST config
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # must be set as secret in Actions
OPENAI_MODEL = os.environ.get("SCI_BOT_MODEL", "gpt-3.5-turbo")
OPENAI_TIMEOUT = int(os.environ.get("SCI_BOT_OPENAI_TIMEOUT", 30))

# Optional config overrides (config.json)
CONFIG_PATH = os.environ.get("SCI_BOT_CONFIG", "config.json")

def load_config_overrides():
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            if "ALLOWED_DOMAINS" in cfg:
                ALLOWED_DOMAINS.update(set(cfg["ALLOWED_DOMAINS"]))
            if "BLOCK_KEYWORDS" in cfg:
                BLOCK_KEYWORDS.update(set(cfg["BLOCK_KEYWORDS"]))
            if "RSS_SOURCES" in cfg and isinstance(cfg["RSS_SOURCES"], dict):
                RSS_SOURCES.update(cfg["RSS_SOURCES"])
            if "RECENCY_DAYS" in cfg:
                global RECENCY_DAYS
                RECENCY_DAYS = int(cfg["RECENCY_DAYS"])
    except Exception as ex:
        logging.warning("Config override failed: %s", ex)

RECENCY_DAYS = int(os.environ.get("SCI_BOT_RECENCY_DAYS", 7))

# ---------------- Helpers ----------------

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: max(0, n - 1)].rstrip() + "…"

def domain_hashtag(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        dom = f"{ext.domain}".capitalize()
        return f"#{dom}"
    except Exception:
        return ""

def is_preprint(url: str) -> bool:
    return "arxiv.org" in url

def allowed_by_policy(url: str, title: str, body: str, published: Optional[datetime]) -> bool:
    try:
        ext = tldextract.extract(url)
        domain = ".".join([d for d in [ext.domain, ext.suffix] if d])
    except Exception:
        domain = ""
    if domain not in ALLOWED_DOMAINS:
        return False
    lt = (title + " " + body[:400]).lower()
    if any(k in lt for k in BLOCK_KEYWORDS):
        return False
    if published and published < datetime.now(timezone.utc) - timedelta(days=RECENCY_DAYS):
        return False
    return True

# ---------------- Storage (SQLite) ----------------

def ensure_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS posts(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT UNIQUE,
        title TEXT,
        posted_at TEXT)""")
    conn.commit()
    return conn

def already_posted(conn, url: str) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM posts WHERE url = ?", (url,))
    return cur.fetchone() is not None

def mark_posted(conn, url: str, title: str):
    cur = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO posts(url, title, posted_at) VALUES(?,?,?)",
                (url, title, datetime.now(timezone.utc).isoformat()))
    conn.commit()

# ---------------- Fetching ----------------

def fetch_article_text(url: str, timeout: int = 15) -> str:
    """Fetch and extract main article text; return empty string on failure."""
    try:
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        selectors = ["article", "div.article-body", "div#content", "div.post-content", "main"]
        for sel in selectors:
            el = soup.select_one(sel)
            if el and len(el.get_text(strip=True)) > 200:
                return clean_text(el.get_text(" "))
        # fallback: full text
        return clean_text(soup.get_text(" "))
    except Exception as e:
        logging.debug("fetch_article_text failed for %s: %s", url, e)
        return ""

def gather_items(source_keys: List[str]):
    items = []
    for key in source_keys:
        for feed in RSS_SOURCES.get(key, []):
            try:
                fp = feedparser.parse(feed)
                for e in fp.entries[:30]:
                    title = clean_text(e.get("title", "").strip())
                    link = e.get("link")
                    published = None
                    if hasattr(e, 'published_parsed') and e.published_parsed:
                        published = datetime.fromtimestamp(time.mktime(e.published_parsed), tz=timezone.utc)
                    # store feed summary as fallback
                    summary = e.get("summary", "") or e.get("description", "")
                    if title and link:
                        items.append((title, link, key, published, summary))
            except Exception as ex:
                logging.warning("Feed error %s: %s", feed, ex)
    # de-duplicate
    seen = set()
    uniq = []
    for t, l, k, p, s in items:
        if l not in seen:
            uniq.append((t, l, k, p, s))
            seen.add(l)
    random.shuffle(uniq)
    return uniq

# ---------------- Summarization (OpenAI REST) ----------------

def build_summary(title: str, text: str, url: str, lang: str = "en", preprint: bool = False) -> str:
    """
    Use OpenAI Chat Completions API via REST to return a two-sentence summary.
    Returns empty string on failure.
    """
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY not set.")
        return ""

    prompt = (
        f"You are a concise science editor. Write exactly TWO sentences in {lang}. "
        "Be clear, neutral, and specific. Avoid hype. State what was done/found and under what conditions. "
        "Keep it under 220 characters if possible. "
        "Mention the domain (AI/quantum/fusion/longevity/BMI) when obvious. "
        "If source is a preprint, begin the first sentence with 'Preprint:'.\n"
        f"Title: {title}\nArticle text:\n{text[:6000]}"
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": "You write crisp, factual science summaries in exactly two sentences."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 140,
    }

    try:
        resp = requests.post(f"{OPENAI_API_URL}/chat/completions", headers=headers, json=payload, timeout=OPENAI_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logging.warning("OpenAI request failed: %s", e)
        try:
            logging.debug("OpenAI response text: %s", resp.text)
        except Exception:
            pass
        return ""

    try:
        data = resp.json()
        # robust extraction of content
        content = ""
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            if isinstance(choice.get("message"), dict):
                content = choice["message"].get("content", "")
            else:
                content = choice.get("text", "")
        else:
            content = ""
    except Exception as e:
        logging.warning("Failed to parse OpenAI response: %s -- body: %s", e, resp.text)
        return ""

    if preprint and content and not content.lower().startswith("preprint:"):
        content = "Preprint: " + content

    return truncate(clean_text(content), 240)

# ---------------- Posting (Tweepy) ----------------

def make_twitter_client():
    api_key = os.environ.get("X_API_KEY")
    api_secret = os.environ.get("X_API_SECRET")
    access_token = os.environ.get("X_ACCESS_TOKEN")
    access_secret = os.environ.get("X_ACCESS_TOKEN_SECRET")
    if not all([api_key, api_secret, access_token, access_secret]):
        raise RuntimeError("Missing X/Twitter credentials in env.")
    auth = tweepy.OAuth1UserHandler(api_key, api_secret, access_token, access_secret)
    return tweepy.API(auth)

def craft_tweet(title: str, url: str, summary: str) -> str:
    base = f"{title}\n{summary}\n{url}"
    tags = []
    dh = domain_hashtag(url)
    if dh:
        tags.append(dh)
    tags.append(random.choice(FOCUS_HASHTAGS))
    # reserve space for tags
    reserved = (len(" ") + sum(len(t) + 1 for t in tags))
    tweet = truncate(base, MAX_TWEET_LEN - reserved)
    return tweet + " " + " ".join(tags)

# ---------------- Main routine ----------------

def run(max_posts: int = 10, lang: str = "en", post: bool = False, source_keys: Optional[List[str]] = None, dry_run: bool = False):
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Allow runtime config overrides
    load_config_overrides()

    conn = ensure_db()

    if source_keys is None or source_keys == ["all"]:
        source_keys = list(RSS_SOURCES.keys())

    items = gather_items(source_keys)
    logging.info("Fetched %d candidate items from %d sources", len(items), len(source_keys))

    api = None
    if post and not dry_run:
        api = make_twitter_client()

    posted = 0
    now_ts = time.time()

    for (title, link, key, published, feed_summary) in items:
        if posted >= max_posts:
            break
        if already_posted(conn, link):
            continue

        # Recency check already handled in allowed_by_policy, but skip if missing and too old
        if published and (time.time() - published.timestamp()) > RECENCY_DAYS * 86400:
            continue

        article_text = fetch_article_text(link)
        # fallback to feed summary if article fetch is too short
        if len(article_text) < 300:
            article_text = feed_summary or article_text

        if len(article_text) < 100:
            logging.debug("Skipping %s: insufficient article text", link)
            continue

        if not allowed_by_policy(link, title, article_text, published):
            logging.debug("Filtered by policy: %s", link)
            continue

        preprint = is_preprint(link)

        try:
            summary = build_summary(title, article_text, link, lang=lang, preprint=preprint)
        except Exception as ex:
            logging.warning("Summarization failed for %s: %s", link, ex)
            logging.debug(traceback.format_exc())
            continue

        if not summary:
            logging.debug("Empty summary for %s, skipping", link)
            continue

        t_title = ("[Preprint] " + title) if preprint else title
        tweet = craft_tweet(t_title, link, summary)

        if dry_run or not post:
            print("\n--- TWEET PREVIEW ---\n" + tweet + "\n")
            mark_posted(conn, link, t_title)
            posted += 1
            continue

        try:
            api.update_status(status=tweet)
            mark_posted(conn, link, t_title)
            posted += 1
            logging.info("Posted: %s", truncate(t_title, 80))
            time.sleep(random.uniform(8, 18))
        except Exception as ex:
            logging.error("Posting failed for %s: %s", link, ex)
            logging.debug(traceback.format_exc())
            continue

    logging.info("Done. Posted %d", posted)

# ---------------- CLI ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true", help="Actually post to X/Twitter")
    ap.add_argument("--dry-run", action="store_true", help="Print tweets instead of posting")
    ap.add_argument("--max", type=int, default=10, help="Number of posts (default 10)")
    ap.add_argument("--lang", type=str, default="en", help="Summary language (default en)")
    ap.add_argument("--sources", type=str, default="all", help="Comma-separated source keys (default all)")
    args = ap.parse_args()

    max_posts = max(1, min(15, args.max))
    src = [s.strip() for s in args.sources.split(",") if s.strip()]
    run(max_posts=max_posts, lang=args.lang, post=args.post, source_keys=src, dry_run=args.dry_run)
