import praw
import json
import requests
import os
from datetime import datetime

# --- Initialize Reddit instance ---
reddit = praw.Reddit(
    client_id="3x_HzTZJyV5o85oPFMffGg",
    client_secret="-S9GR9_wDSHYHB93fYRrbQTOjqQWTg",
    user_agent="dynamic_reddit_scraper"
)

def fetch_and_save_json(url, output_dir="restuarants"):
    """Fetch a Reddit post (and comments) using Reddit's .json API and save to file."""
    headers = {"User-Agent": "script:reddit-json:v1.0 (by u/midtowndriverr)"}
    if not url.endswith(".json"):
        url = url.rstrip("/") + "/.json"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Use post ID or title for filename
    post_id = url.split("/")[-3]
    file_path = os.path.join(output_dir, f"{post_id}.json")

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved Reddit post and comments → {file_path}")
    return file_path


def dynamic_reddit_scraper(subreddit_name="news", keyword=None, limit=10):
    """Automatically fetch multiple Reddit posts dynamically and save to JSON."""
    subreddit = reddit.subreddit(subreddit_name)
    posts = subreddit.search(keyword, limit=limit) if keyword else subreddit.hot(limit=limit)

    saved_files = []
    for submission in posts:
        print(f"📥 Fetching: {submission.title}")
        post_url = f"https://www.reddit.com{submission.permalink}"
        saved_file = fetch_and_save_json(post_url)
        saved_files.append(saved_file)

    print(f"\n✅ Finished! Saved {len(saved_files)} posts from r/{subreddit_name}.")
    return saved_files


# --- MAIN ---
if __name__ == "__main__":
    subreddit_name = input("Enter subreddit name (default=news): ") or "news"
    keyword = input("Enter keyword to search (or leave blank for hot posts): ") or None
    limit = int(input("How many posts to fetch? (default=10): ") or 10)

    dynamic_reddit_scraper(subreddit_name, keyword, limit)