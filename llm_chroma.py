import chromadb
import json, os

# --- initialize chroma client ---
client = chromadb.CloudClient(
    api_key='ck-Bz6vWE8LH4mrmKx2dfreGSzgiikFBKhm7BHNHivkDvvw',
    tenant='6db89e03-6466-4af4-ad1c-d8237a75efa7',
    database='places'
)

# --- create (or get) a collection ---
collection = client.get_or_create_collection(name="cleaned_posts")

# ------------------ Load JSON Files ------------------
data_dir = "llm_extracted"  # folder containing your JSON files

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