import chromadb

# If you used PersistentClient previously:
client = chromadb.CloudClient(
  api_key='ck-Bz6vWE8LH4mrmKx2dfreGSzgiikFBKhm7BHNHivkDvvw',
  tenant='6db89e03-6466-4af4-ad1c-d8237a75efa7',
  database='places'
)

# Or if you used in-memory:
# client = chromadb.Client()

collection = client.get_collection("reddit_posts")
count = collection.count()
print(f"Total documents in collection: {count}")
results = collection.peek(5)  # peek first 5 documents

for doc, meta in zip(results['documents'], results['metadatas']):
    print("Title:", meta.get("title"))
    print("Subreddit:", meta.get("subreddit"))
    print("Author:", meta.get("author"))
    print("Content snippet:\n", doc[:1000], "...\n")  # first 500 chars
