import json
import os
import re
from pathlib import Path
from typing import List, Dict
import chromadb
from huggingface_hub import InferenceClient

# ============================================================================
# CHROMADB CONFIGURATION
# ============================================================================

client = chromadb.CloudClient(
    api_key='ck-Bz6vWE8LH4mrmKx2dfreGSzgiikFBKhm7BHNHivkDvvw',
    tenant='6db89e03-6466-4af4-ad1c-d8237a75efa7',
    database='places'
)

collection = client.get_or_create_collection(name="cleaned_posts")


# ============================================================================
# STEP 1: DATA CLEANING - Extract Reddit Data
# ============================================================================

def extract_reddit_data(reddit_json: List[Dict]) -> Dict:
    """Extract post and comments from Reddit API JSON"""

    post_listing = reddit_json[0]['data']['children'][0]['data']

    post_id = post_listing['id']
    title = post_listing['title']
    content = post_listing['selftext']
    subreddit = post_listing['subreddit']

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

    return output_path


# ============================================================================
# STEP 2: LLM EXTRACTION - Initialize the LLM Model
# ============================================================================

print("🔄 Loading Mistral model...")

# Using Hugging Face Inference API (FREE, no local GPU needed!)
HF_API_KEY = "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"
llm_client = InferenceClient(token=HF_API_KEY)

# Choose model (all FREE):
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # Best quality
# MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"  # Faster, smaller
# MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"  # Alternative good option

print(f"✅ Using model: {MODEL_NAME}\n")

# ============================================================================
# STEP 3: Extraction Prompt Template
# ============================================================================

EXTRACTION_PROMPT = """You are a travel information extraction expert. Extract place information from this Reddit discussion.

Reddit Post:
Title: {title}
Subreddit: {subreddit}
Content: {content}
Comments: {body_chunk}

Your task: Extract ALL mentions of places, activities, restaurants, and recommendations.

For EACH place/location/activity mentioned, create an entry with:
- theme: (choose one: "things_to_do", "hidden_gems", "restaurants", "practical_tips", "warnings")
- location: (specific address/neighborhood if mentioned, otherwise city name)
- places: (name of the place, restaurant, or activity)
- notes: (brief 1-2 sentence summary of what people said about it)
- sentiment: (choose one: "great", "good", "okay", "bad", "mixed")

Return ONLY a JSON array like this:
[
  {{
    "theme": "things_to_do",
    "location": "Philadelphia, PA",
    "places": "City Hall Observation Deck",
    "notes": "Offers amazing views of the city. The tour is 2+ hours with only last 20 minutes on the deck.",
    "sentiment": "good"
  }},
  {{
    "theme": "hidden_gems",
    "location": "City Hall basement",
    "places": "Old boiler room",
    "notes": "Fantastic Jules Verne-style boiler room from 1876. Used to showcase technology for the centennial.",
    "sentiment": "great"
  }}
]

Important:
- Extract EVERY place, restaurant, activity, or location mentioned
- If no specific place names, extract general recommendations (e.g., "walk around downtown")
- Keep notes concise but informative
- Use exact names when available
- Return ONLY valid JSON array, no other text or explanation"""


# ============================================================================
# STEP 4: Chunk the body into manageable pieces
# ============================================================================

def chunk_text(text: str, max_length: int = 3000) -> List[str]:
    """
    Split long text into chunks that fit in model context.
    Split by paragraphs/newlines to keep context intact.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= max_length:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================================
# STEP 5: Extract with LLM (handles long body by chunking)
# ============================================================================

def extract_with_mistral(title: str, subreddit: str, content: str, body: str) -> List[Dict]:
    """
    Extract structured place information using Mistral LLM.
    Handles long 'body' by processing it in chunks.
    """
    all_extractions = []

    # Chunk the long body text
    body_chunks = chunk_text(body, max_length=3000)

    print(f"  📝 Processing {len(body_chunks)} chunk(s)...")

    for i, body_chunk in enumerate(body_chunks):
        print(f"\n    --- Chunk {i + 1}/{len(body_chunks)} ---")

        try:
            # Create prompt
            prompt = EXTRACTION_PROMPT.format(
                title=title,
                subreddit=subreddit,
                content=content[:500] if content else "",  # Limit content
                body_chunk=body_chunk
            )

            # Call Hugging Face API using CHAT COMPLETION
            messages = [
                {"role": "user", "content": prompt}
            ]

            print(f"    🤖 Calling LLM API...")

            response = llm_client.chat_completion(
                messages=messages,
                model=MODEL_NAME,
                max_tokens=1500,
                temperature=0.1
            )

            # Extract the response text
            response_text = response.choices[0].message.content.strip()

            # DEBUG: Print what we got
            print(f"    🔍 Raw response (first 300 chars):")
            print(f"    {response_text[:300]}...")

            # Remove markdown code blocks if present
            response_text = re.sub(r'```json\s*|\s*```', '', response_text)
            response_text = response_text.strip()

            # Find JSON array in response
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(0)
                print(f"    ✓ Found JSON array")
            else:
                print(f"    ⚠️  No JSON array found in response")
                print(f"    Full response: {response_text}")
                continue

            # Parse JSON
            chunk_extractions = json.loads(response_text)

            if isinstance(chunk_extractions, list):
                all_extractions.extend(chunk_extractions)
                print(f"    ✅ Successfully extracted {len(chunk_extractions)} place(s) from this chunk")
            else:
                print(f"    ⚠️  Response is not a list")

        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parse error: {e}")
            print(f"    Attempted to parse: {response_text[:300]}...")
        except Exception as e:
            print(f"    ❌ Error: {e}")

    # Remove duplicates based on place name
    unique_extractions = []
    seen_places = set()

    for item in all_extractions:
        place_name = item.get('places', '').lower().strip()
        if place_name and place_name not in seen_places:
            seen_places.add(place_name)
            unique_extractions.append(item)

    print(f"\n  🎯 Total unique places after deduplication: {len(unique_extractions)}")

    return unique_extractions


# ============================================================================
# STEP 6: Process input JSON and save output
# ============================================================================

def process_reddit_json(input_file: str, output_file: str):
    """
    Read input JSON, extract places with LLM, save to output JSON.

    Input format: {subreddit, id, title, content, body}
    Output format: [{subreddit_name, theme, location, places, notes, sentiment}, ...]
    """
    print(f"\n{'=' * 60}")
    print(f"📂 Processing: {os.path.basename(input_file)}")
    print(f"{'=' * 60}")

    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    subreddit = data.get('subreddit', '')
    post_id = data.get('id', '')
    title = data.get('title', '')
    content = data.get('content', '')
    body = data.get('body', '')

    print(f"📍 Subreddit: r/{subreddit}")
    print(f"📝 Title: {title[:80]}{'...' if len(title) > 80 else ''}")
    print(f"💬 Body length: {len(body)} characters")

    # Extract with LLM
    extractions = extract_with_mistral(title, subreddit, content, body)

    # Add subreddit_name to each extraction
    output_data = []
    for item in extractions:
        output_data.append({
            "subreddit_name": subreddit,
            "theme": item.get("theme", ""),
            "location": item.get("location", ""),
            "places": item.get("places", ""),
            "notes": item.get("notes", ""),
            "sentiment": item.get("sentiment", "")
        })

    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Extracted {len(output_data)} unique place(s)")
    print(f"💾 Saved to: {os.path.basename(output_file)}")

    return output_data


# ============================================================================
# STEP 7: Batch process all JSON files in a directory
# ============================================================================

def process_directory(input_dir: str, output_dir: str):
    """Process all JSON files in input_dir"""

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_files = list(input_path.glob('*.json'))

    if not json_files:
        print(f"❌ No JSON files found in {input_dir}")
        return

    print(f"\n🚀 Found {len(json_files)} file(s) to process\n")

    success_count = 0
    total_places = 0

    for json_file in json_files:
        try:
            output_file = output_path / f"{json_file.stem}_extracted.json"

            extractions = process_reddit_json(str(json_file), str(output_file))

            success_count += 1
            total_places += len(extractions)

        except Exception as e:
            print(f"\n✗ Error processing {json_file.name}: {e}\n")

    print(f"\n{'=' * 60}")
    print(f"🎉 COMPLETE!")
    print(f"✅ Successfully processed: {success_count}/{len(json_files)} file(s)")
    print(f"📍 Total places extracted: {total_places}")
    print(f"📁 Output directory: {output_path}")
    print(f"{'=' * 60}\n")

    return output_path


# ============================================================================
# STEP 8: CHROMADB UPLOAD - Upload extracted data to ChromaDB
# ============================================================================

def upload_to_chromadb(data_dir: str):
    """Upload extracted JSON files to ChromaDB"""

    print(f"\n{'=' * 60}")
    print("📤 UPLOADING TO CHROMADB")
    print(f"{'=' * 60}\n")

    success_count = 0
    error_count = 0
    total_places = 0

    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check if data is a list
            if not isinstance(data, list):
                print(f"❌ Skipped {filename}: Data is not a list")
                error_count += 1
                continue

            if len(data) == 0:
                print(f"⚠️  Skipped {filename}: Empty list")
                error_count += 1
                continue

            # Process each place entry in the JSON file
            for idx, entry in enumerate(data):
                try:
                    # Extract fields
                    subreddit = entry.get("subreddit_name", "unknown")
                    theme = entry.get("theme", "")
                    location = entry.get("location", "")
                    places = entry.get("places", "")
                    notes = entry.get("notes", "")
                    sentiment = entry.get("sentiment", "")

                    # Create a unique ID for this entry
                    place_id = f"{filename.replace('.json', '')}_{idx}"

                    # Create document text for searching
                    full_document = f"""SUBREDDIT: {subreddit}
THEME: {theme}
LOCATION: {location}
PLACE: {places}
NOTES: {notes}
SENTIMENT: {sentiment}"""

                    # Add to ChromaDB
                    collection.add(
                        ids=[place_id],
                        documents=[full_document],
                        metadatas=[{
                            "subreddit_name": subreddit,
                            "theme": theme,
                            "location": location,
                            "places": places,
                            "notes": notes,
                            "sentiment": sentiment,
                            "source_file": filename
                        }]
                    )

                    total_places += 1

                except Exception as e:
                    print(f"❌ Error processing entry {idx} in {filename}: {e}")
                    continue

            print(f"✅ Processed {filename}: {len(data)} places")
            success_count += 1

        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error in {filename}: {e}")
            error_count += 1
        except Exception as e:
            print(f"❌ Error in {filename}: {type(e).__name__} - {e}")
            error_count += 1

    print(f"\n{'=' * 60}")
    print(f"✅ Successfully processed: {success_count} files")
    print(f"❌ Errors: {error_count} files")
    print(f"📍 Total places added: {total_places}")
    print(f"{'=' * 60}")
    print(f"\nTotal documents in collection: {collection.count()}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":

    # ====================
    # STEP 1: Clean Reddit JSON files
    # ====================
    print("\n" + "=" * 60)
    print("STEP 1: CLEANING REDDIT JSON FILES")
    print("=" * 60 + "\n")

    cleaned_dir = convert_reddit_json_files(
        input_dir='/Users/santhiyatheanraj/PycharmProjects/HCI/reddit_jsons/',
        output_dir='/Users/santhiyatheanraj/PycharmProjects/HCI/extracted_json/'
    )

    # ====================
    # STEP 2: Extract with LLM
    # ====================
    print("\n" + "=" * 60)
    print("STEP 2: EXTRACTING WITH LLM")
    print("=" * 60 + "\n")

    extracted_dir = process_directory(
        input_dir='/Users/santhiyatheanraj/PycharmProjects/HCI/extracted_json/',
        output_dir='/Users/santhiyatheanraj/PycharmProjects/HCI/llm_extracted/'
    )

    # ====================
    # STEP 3: Upload to ChromaDB
    # ====================
    upload_to_chromadb('/Users/santhiyatheanraj/PycharmProjects/HCI/llm_extracted/')

    # ====================
    # EXAMPLE: See what extracted data looks like
    # ====================
    print("\n📋 SAMPLE OUTPUT:")
    output_path = Path('/Users/santhiyatheanraj/PycharmProjects/HCI/llm_extracted/')
    sample_files = list(output_path.glob('*_extracted.json'))

    if sample_files:
        with open(sample_files[0], 'r') as f:
            sample_data = json.load(f)
            if sample_data:
                print(json.dumps(sample_data[:3], indent=2))  # Show first 3 entries
            else:
                print("No data extracted in this file.")
    else:
        print("No output files found yet.")