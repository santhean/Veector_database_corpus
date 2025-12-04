import json,os
from pathlib import Path
from typing import List, Dict
import chromadb

client = chromadb.CloudClient(
  api_key='ck-Bz6vWE8LH4mrmKx2dfreGSzgiikFBKhm7BHNHivkDvvw',
  tenant='6db89e03-6466-4af4-ad1c-d8237a75efa7',
  database='places'
)

collection = client.get_or_create_collection(name="cleaned_posts")
data_dir = "extracted_json"  # folder containing your Reddit JSON files


def extract_reddit_data(reddit_json: List[Dict]) -> Dict:
    """Extract post and comments from Reddit API JSON"""

    post_listing = reddit_json[0]['data']['children'][0]['data']

    post_id = post_listing['id']
    title = post_listing['title']
    content = post_listing['selftext']
    subreddit=post_listing['subreddit']

    comments_listing = reddit_json[1]['data']['children']
    all_comments = []

    def extract_comments_recursive(comment_list):
        for item in comment_list:
            if item['kind'] != 't1':
                continue

            comment_data = item['data']

            if comment_data.get('body') in ['[deleted]', '[removed]', None]:
                continue

            all_comments.append(comment_data['body'])

            replies = comment_data.get('replies')
            if replies and isinstance(replies, dict):
                reply_children = replies['data']['children']
                extract_comments_recursive(reply_children)

    extract_comments_recursive(comments_listing)

    body = '\n\n'.join(all_comments)

    return {
        'subreddit': subreddit,
        'id': post_id,
        'title': title,
        'content': content,
        'body': body
    }


def convert_reddit_json_files(input_dir: str = 'data/reddit_raw',
                              output_dir: str = 'data/reddit_cleaned'):
    """
    Main function to convert all Reddit JSON files
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} files to process\n")

    success_count = 0
    error_count = 0

    for json_file in json_files:
        try:
            # Read original
            with open(json_file, 'r', encoding='utf-8') as f:
                reddit_data = json.load(f)

            # Extract
            extracted = extract_reddit_data(reddit_data)

            # Save cleaned version
            output_file = output_path / f"{json_file.stem}_cleaned.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(extracted, f, indent=2, ensure_ascii=False)

            print(f"✓ {json_file.name}")
            success_count += 1

        except Exception as e:
            print(f"✗ {json_file.name}: {e}")
            error_count += 1

    print(f"\n{'=' * 50}")
    print(f"Success: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Output directory: {output_path}")


if __name__ == "__main__":
    # Single file
    # process_reddit_json_to_new_file('reddit_post.json', 'reddit_post_cleaned.json')

    # All files in directory
    convert_reddit_json_files(
        input_dir='/Users/santhiyatheanraj/PycharmProjects/HCI/reddit_jsons/',
        output_dir='/Users/santhiyatheanraj/PycharmProjects/HCI/extracted_json/'
    )


    # ------------------ Utility function to split large text ------------------
    def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
        """Split text into smaller chunks (approx chunk_size characters each)."""
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


    # ------------------ Process & Upload JSON files ------------------
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract fields
            post_id = data.get("id", filename.replace(".json", ""))
            title = data.get("title", "")
            content = data.get("content", "")
            body = data.get("body", "")

            # Assemble full text
            full_doc = f"TITLE:\n{title}\n\nCONTENT:\n{content}\n\nCOMMENTS:\n{body}"

            # Split into chunks to respect Chroma Cloud size limits
            chunks = chunk_text(full_doc, chunk_size=5000)

            for i, chunk in enumerate(chunks):
                collection.add(
                    ids=[f"{post_id}_{i}"],  # unique ID per chunk
                    documents=[chunk],
                    metadatas=[{
                        "subreddit": data.get("subreddit", ""),
                        "title": title,  # keep metadata small
                        "chunk_index": i
                    }]
                )

            print(f"✓ Uploaded {filename} in {len(chunks)} chunk(s)")

        except Exception as e:
            print(f"✗ Error processing {filename}: {e}")

    print("\n✅ All files processed and uploaded safely.")

