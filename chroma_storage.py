import chromadb
import json,os

# --- initialize chroma client ---
def extract_comments(comment_list):
    all_text = []
    for c in comment_list:
        if c["kind"] != "t1":
            continue
        body = c["data"].get("body", "")
        all_text.append(body)
        # Recursively check replies
        replies = c["data"].get("replies")
        if replies and isinstance(replies, dict):
            all_text.append(extract_comments(replies["data"]["children"]))
    return "\n".join(all_text)

client = chromadb.CloudClient(
  api_key='ck-Bz6vWE8LH4mrmKx2dfreGSzgiikFBKhm7BHNHivkDvvw',
  tenant='6db89e03-6466-4af4-ad1c-d8237a75efa7',
  database='places'
)

# --- create (or get) a collection ---
collection = client.get_or_create_collection(name="reddit_posts")
# ------------------ Load JSON Files ------------------
data_dir = "restuarants"  # folder containing your Reddit JSON files

for filename in os.listdir(data_dir):
    if not filename.endswith(".json"):
        continue

    filepath = os.path.join(data_dir, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        post_info = data[0]["data"]["children"][0]["data"]
        post_id = post_info.get("id", filename.replace(".json", ""))
        title = post_info.get("title", "")
        selftext = post_info.get("selftext", "")

        # Extract all comments from JSON
        comments = ""
        if len(data) > 1:
            comments = extract_comments(data[1]["data"]["children"])

        # Combine post + comments
        content = f"{title}\n\n{selftext}\n\nComments:\n{comments}"
        # Add to Chroma Cloud
        collection.add(
            ids=[post_id],
            documents=[content],
            metadatas=[{
                "title": title,
                "subreddit": post_info.get("subreddit", ""),
                "author": post_info.get("author", ""),
                "url": f"https://www.reddit.com{post_info.get('permalink', '')}",
                "created_utc": post_info.get("created_utc")
            }]
        )
        print(f"✅ Added: {title[:60]}...")

        print(f" Added: {title[:60]}...")

    except Exception as e:
        print(f"Skipped {filename}: {e}")

print("\nAll Reddit JSON files have been loaded into Chroma Cloud.")
