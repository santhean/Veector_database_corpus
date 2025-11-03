import chromadb
import requests
import json


def initialize_chroma():
    """Initialize ChromaDB client."""
    client = chromadb.CloudClient(
        api_key='ck-Bz6vWE8LH4mrmKx2dfreGSzgiikFBKhm7BHNHivkDvvw',
        tenant='6db89e03-6466-4af4-ad1c-d8237a75efa7',
        database='places'
    )
    return client


def search_reddit_posts(collection, query: str, n_results: int = 5):
    """Search ChromaDB for relevant Reddit posts."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results


def format_context(results):
    """Format search results into context for LLM."""
    contexts = []

    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        # Truncate long documents
        content = doc[:800] if len(doc) > 800 else doc

        context = f"""
[Post {i + 1}]
Title: {metadata['title']}
Subreddit: r/{metadata['subreddit']}
URL: {metadata['url']}

{content}
"""
        contexts.append(context)

    return "\n".join(contexts)


def generate_answer_ollama(query: str, context: str, model: str = "llama2"):
    """Generate answer using local Ollama LLM."""
    prompt = f"""Based on the following Reddit posts, answer the user's question. Be concise and cite which post(s) you're referencing.

Reddit Context:
{context}

Question: {query}

Answer:"""

    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 400
                }
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: Ollama returned status {response.status_code}"

    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to Ollama. Make sure it's running: 'ollama serve'"
    except Exception as e:
        return f"❌ Error: {e}"


def generate_answer_huggingface(query: str, context: str):
    """Generate answer using HuggingFace Inference API (free tier)."""
    import os

    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

    prompt = f"""<s>[INST] Based on these Reddit posts, answer the question concisely:

{context}

Question: {query} [/INST]"""

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.7}},
            timeout=30
        )

        if response.status_code == 200:
            return response.json()[0]['generated_text'].split('[/INST]')[-1].strip()
        else:
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        return f"❌ Error: {e}"


def rag_query(query: str, n_results: int = 5, llm_provider: str = "ollama"):
    """
    Main RAG pipeline.

    Args:
        query: User's search query
        n_results: Number of documents to retrieve
        llm_provider: 'ollama' or 'huggingface'
    """
    print(f"🔍 Searching for: '{query}'\n")

    # Initialize ChromaDB
    chroma_client = initialize_chroma()
    collection = chroma_client.get_collection(name="reddit_posts")

    # Step 1: Retrieve relevant documents
    print("📥 Retrieving relevant Reddit posts...")
    results = search_reddit_posts(collection, query, n_results)

    if not results['documents'][0]:
        print("❌ No relevant posts found.")
        return None

    print(f"✅ Found {len(results['documents'][0])} relevant posts\n")

    # Step 2: Display retrieved posts
    print("=" * 60)
    print("RETRIEVED POSTS:")
    print("=" * 60)
    for i, metadata in enumerate(results['metadatas'][0]):
        print(f"\n{i + 1}. {metadata['title'][:70]}...")
        print(f"   r/{metadata['subreddit']} | {metadata['url']}")

    # Step 3: Format context
    context = format_context(results)

    # Step 4: Generate answer with LLM
    print("\n" + "=" * 60)
    print(f"GENERATING ANSWER (using {llm_provider})...")
    print("=" * 60 + "\n")

    if llm_provider == "ollama":
        answer = generate_answer_ollama(query, context)
    elif llm_provider == "huggingface":
        answer = generate_answer_huggingface(query, context)
    else:
        answer = "❌ Invalid LLM provider. Choose 'ollama' or 'huggingface'"

    print("🤖 Answer:")
    print(answer)
    print("\n" + "=" * 60)

    return {
        "answer": answer,
        "sources": results['metadatas'][0]
    }


def interactive_mode(llm_provider: str = "ollama"):
    """Run RAG in interactive mode."""
    print("=" * 60)
    print("Reddit RAG Search System")
    print(f"LLM Provider: {llm_provider}")
    print("=" * 60)
    print("Type your question or 'quit' to exit\n")

    while True:
        query = input("❓ Your question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        if not query:
            continue

        try:
            rag_query(query, llm_provider=llm_provider)
            print("\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


# Example usage
if __name__ == "__main__":
    # Choose your LLM provider:
    # - "ollama" (free, local, requires: ollama pull llama2)
    # - "huggingface" (free, online, requires HF_TOKEN env var)

    # Single query example
    rag_query("What are good restaurants in Paris?", llm_provider="ollama")

    # Or run interactive mode
    # interactive_mode(llm_provider="ollama")