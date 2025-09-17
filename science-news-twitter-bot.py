"""
Science News Twitter Bot (Upgraded)

Adds:
- Domain allowlist (trusted science sources)
- Keyword blocklist (exclude fringe)
- Preprint labeling for arXiv
- Recency filter (skip >7 days old)

Posts 10 curated science/tech headlines daily with 2‑sentence summaries and source links.
Focus areas: AI, quantum, fusion, longevity/biotech, brain‑machine interfaces.
"""
from __future__ import annotations
import os, sys, re, time, json, textwrap, argparse, sqlite3, random, logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple

import feedparser
import requests
from bs4 import BeautifulSoup
import tweepy
from dotenv import load_dotenv
import tldextract
import openai

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

# ---------- Helpers ----------

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

# ---------- Storage ----------

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

# ---------- Filters ----------

def is_allowed(url: str, title: str, body: str, published: datetime | None) -> bool:
    ext = tldextract.extract(url)
    domain = ".".join([d for d in [ext.domain, ext.suffix] if d])
    if domain not in ALLOWED_DOMAINS:
        return False
    lt = (title + " " + body[:400]).lower()
    if any(k in lt for k in BLOCK_KEYWORDS):
        return False
    if published and published < datetime.now(timezone.utc) - timedelta(days=7):
        return False
    return True

# ---------- Fetching ----------

def fetch_article_text(url: str, timeout: int = 15) -> str:
    try:
        headers = {"User-Agent": USER_AGENT}
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        selectors = ["article","div.article-body","div#content","div.post-content","main"]
        for sel in selectors:
            el = soup.select_one(sel)
            if el and len(el.get_text(strip=True)) > 200:
                return clean_text(el.get_text(" "))
        return clean_text(soup.get_text(" "))
    except Exception:
        return ""

def gather_items(source_keys: List[str]):
    items = []
    for key in source_keys:
        for feed in RSS_SOURCES.get(key, []):
            try:
                fp = feedparser.parse(feed)
                for e in fp.entries[:20]:
                    title = clean_text(e.get("title", "").strip())
                    link = e.get("link")
                    published = None
                    if hasattr(e, 'published_parsed') and e.published_parsed:
                        published = datetime.fromtimestamp(time.mktime(e.published_parsed), tz=timezone.utc)
                    if title and link:
                        items.append((title, link, key, published))
            except Exception as ex:
                logging.warning("Feed error %s: %s", feed, ex)
    seen = set()
    uniq = []
    for t, l, k, p in items:
        if l not in seen:
            uniq.append((t, l, k, p))
            seen.add(l)
    random.shuffle(uniq)
    return uniq

# ---------- Summarization ----------
def build_summary(title: str, text: str, url: str, lang: str = "en") -> str:
    prompt = f"""You are a concise science editor. Write exactly TWO sentences in {lang}. Be clear, neutral, and specific. Keep it under 220 characters if possible. Mention the domain (AI/quantum/fusion/longevity/BMI) when obvious, avoid hype. If source is a preprint, begin with 'Preprint:'.\nTitle: {title}\nArticle text:\n{text[:6000]}"""
    # Use the openai module (classic ChatCompletion path) to avoid client constructor issues.
    resp = openai.ChatCompletion.create(
        model="gpt-5.1-mini",
        messages=[
            {"role": "system", "content": "You write crisp, factual science summaries in exactly two sentences."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=140,
    )
    return truncate(clean_text(resp.choices[0].message.content), 240)

# ---------- Posting ----------
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
    tweet = truncate(base, MAX_TWEET_LEN - (len(" ") + sum(len(t) + 1 for t in tags)))
    return tweet + " " + " ".join(tags)

# ---------- Main ----------
def run(max_posts=10, lang="en", post=False, source_keys=None, dry_run=False):
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    # Use the openai module and read API key from the environment
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    conn = ensure_db()

    if source_keys is None or source_keys == ["all"]:
        source_keys = list(RSS_SOURCES.keys())

    items = gather_items(source_keys)
    logging.info("Fetched %d candidate items", len(items))

    api = make_twitter_client() if post and not dry_run else None

    posted = 0
    for (title, link, key, published) in items:
        if posted >= max_posts:
            break
        if already_posted(conn, link):
            continue

        article_text = fetch_article_text(link)
        if len(article_text) < 300:
            continue

        if not is_allowed(link, title, article_text, published):
            continue

        try:
            summary = build_summary(title, article_text, link, lang)
        except Exception as ex:
            logging.warning("Summarization failed: %s", ex)
            continue

        if "arxiv.org" in link:
            title = "[Preprint] " + title

        tweet = craft_tweet(title, link, summary)

        if dry_run or not post:
            print("\n--- TWEET PREVIEW ---\n" + tweet + "\n")
            mark_posted(conn, link, title)
            posted += 1
            continue

        try:
            api.update_status(status=tweet)
            mark_posted(conn, link, title)
            posted += 1
            logging.info("Posted: %s", truncate(title, 80))
            time.sleep(random.uniform(8, 18))
        except Exception as ex:
            logging.error("Posting failed: %s", ex)
            continue

    logging.info("Done. Posted %d", posted)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max", type=int, default=10)
    ap.add_argument("--lang", type=str, default="en")
    ap.add_argument("--sources", type=str, default="all")
    args = ap.parse_args()

    max_posts = max(1, min(15, args.max))
    src = [s.strip() for s in args.sources.split(",") if s.strip()]
    run(max_posts=max_posts, lang=args.lang, post=args.post, source_keys=src, dry_run=args.dry_run)
