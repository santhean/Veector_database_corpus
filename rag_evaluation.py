"""
RAG Evaluation System for Places Finder
Evaluates retrieval quality, answer quality, and overall performance
"""

import os
from typing import TypedDict, Annotated, List, Dict
from groq import Groq
import chromadb
import json
from datetime import datetime

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
GROQ_API_KEY = "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"
groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================================
# EVALUATION SCHEMAS
# ============================================================================

class RetrievalRelevanceGrade(TypedDict):
    """Grade the relevance of retrieved documents to the question"""
    explanation: Annotated[str, ..., "Explain your reasoning for the score"]
    relevant: Annotated[bool, ..., "True if retrieved documents are relevant"]
    relevance_score: Annotated[int, ..., "Score from 1-5 for how relevant the documents are"]


class AnswerQualityGrade(TypedDict):
    """Grade the quality of the generated answer"""
    explanation: Annotated[str, ..., "Explain your reasoning"]
    is_grounded: Annotated[bool, ..., "True if answer is based on retrieved facts"]
    is_helpful: Annotated[bool, ..., "True if answer addresses the question"]
    hallucination_detected: Annotated[bool, ..., "True if answer contains made-up information"]
    quality_score: Annotated[int, ..., "Overall quality score 1-5"]


class AnswerRelevanceGrade(TypedDict):
    """Grade whether the answer actually addresses the question"""
    explanation: Annotated[str, ..., "Explain your reasoning"]
    addresses_question: Annotated[bool, ..., "True if answer addresses the user's question"]
    completeness_score: Annotated[int, ..., "How complete is the answer (1-5)"]


class ContextPrecisionGrade(TypedDict):
    """Grade the precision of retrieved context"""
    explanation: Annotated[str, ..., "Explain your reasoning"]
    precision_score: Annotated[float, ..., "Ratio of relevant to total retrieved docs (0-1)"]
    num_relevant: Annotated[int, ..., "Number of relevant documents"]
    num_total: Annotated[int, ..., "Total number of documents retrieved"]


# ============================================================================
# EVALUATION PROMPTS
# ============================================================================

RETRIEVAL_RELEVANCE_PROMPT = """You are a teacher grading a quiz. You will be given a QUESTION and a set of FACTS (retrieved documents).

Grade Criteria:
1. Identify FACTS that are completely unrelated to the QUESTION
2. If facts contain ANY keywords or semantic meaning related to the question, consider them relevant
3. It's OK if facts have SOME unrelated information as long as criterion 2 is met

Relevance Definition:
- relevant = True: FACTS contain keywords or semantic meaning related to QUESTION
- relevant = False: FACTS are completely unrelated to QUESTION

Provide:
1. A step-by-step explanation of your reasoning
2. A boolean 'relevant' value
3. A relevance_score from 1-5 (1=not relevant at all, 5=highly relevant)

Respond in JSON format with keys: explanation, relevant, relevance_score"""

ANSWER_QUALITY_PROMPT = """You are evaluating the quality of an AI-generated answer about places/restaurants.

You will be given:
- QUESTION: The user's question
- RETRIEVED_FACTS: Documents retrieved from the database
- ANSWER: The AI's generated answer

Evaluate:
1. is_grounded: Is the answer based on the retrieved facts? (no made-up places)
2. is_helpful: Does it help the user find places?
3. hallucination_detected: Does it mention places NOT in the retrieved facts?
4. quality_score: Overall quality (1-5)

Respond in JSON format with keys: explanation, is_grounded, is_helpful, hallucination_detected, quality_score"""

ANSWER_RELEVANCE_PROMPT = """You are evaluating whether an answer addresses the user's question.

You will be given:
- QUESTION: What the user asked
- ANSWER: The AI's response

Evaluate:
1. addresses_question: Does the answer actually address what was asked?
2. completeness_score: How complete is the answer? (1-5)
   - 1: Doesn't address the question
   - 3: Partially addresses it
   - 5: Fully addresses all aspects

Respond in JSON format with keys: explanation, addresses_question, completeness_score"""

CONTEXT_PRECISION_PROMPT = """You are evaluating the precision of document retrieval.

You will be given:
- QUESTION: The user's question
- DOCUMENTS: List of retrieved documents (numbered)

For EACH document, determine if it's relevant to answering the question.

Provide:
1. precision_score: ratio of relevant docs to total docs (0.0 to 1.0)
2. num_relevant: count of relevant documents
3. num_total: total documents retrieved
4. explanation: which documents were relevant and why

Respond in JSON format with keys: explanation, precision_score, num_relevant, num_total"""


# ============================================================================
# EVALUATOR FUNCTIONS
# ============================================================================

def call_groq_for_eval(system_prompt: str, user_content: str, model: str = "llama-3.3-70b-versatile") -> dict:
    """Helper function to call Groq API for evaluation"""
    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        response_text = completion.choices[0].message.content
        return json.loads(response_text)

    except Exception as e:
        print(f"Error in Groq API call: {e}")
        return None


def evaluate_retrieval_relevance(question: str, documents: List[str]) -> RetrievalRelevanceGrade:
    """Evaluate if retrieved documents are relevant to the question"""

    doc_string = "\n\n".join([f"Document {i + 1}: {doc}" for i, doc in enumerate(documents)])

    user_content = f"""QUESTION: {question}

FACTS:
{doc_string}

Grade the relevance of these facts to the question."""

    result = call_groq_for_eval(RETRIEVAL_RELEVANCE_PROMPT, user_content)

    if result:
        return RetrievalRelevanceGrade(
            explanation=result.get("explanation", ""),
            relevant=result.get("relevant", False),
            relevance_score=result.get("relevance_score", 1)
        )

    # Fallback
    return RetrievalRelevanceGrade(
        explanation="Evaluation failed",
        relevant=False,
        relevance_score=1
    )


def evaluate_answer_quality(question: str, documents: List[str], answer: str) -> AnswerQualityGrade:
    """Evaluate the quality of the generated answer"""

    doc_string = "\n\n".join([f"Document {i + 1}: {doc}" for i, doc in enumerate(documents)])

    user_content = f"""QUESTION: {question}

RETRIEVED_FACTS:
{doc_string}

ANSWER: {answer}

Evaluate the quality of this answer."""

    result = call_groq_for_eval(ANSWER_QUALITY_PROMPT, user_content)

    if result:
        return AnswerQualityGrade(
            explanation=result.get("explanation", ""),
            is_grounded=result.get("is_grounded", False),
            is_helpful=result.get("is_helpful", False),
            hallucination_detected=result.get("hallucination_detected", True),
            quality_score=result.get("quality_score", 1)
        )

    return AnswerQualityGrade(
        explanation="Evaluation failed",
        is_grounded=False,
        is_helpful=False,
        hallucination_detected=True,
        quality_score=1
    )


def evaluate_answer_relevance(question: str, answer: str) -> AnswerRelevanceGrade:
    """Evaluate if the answer addresses the question"""

    user_content = f"""QUESTION: {question}

ANSWER: {answer}

Does this answer address the question?"""

    result = call_groq_for_eval(ANSWER_RELEVANCE_PROMPT, user_content)

    if result:
        return AnswerRelevanceGrade(
            explanation=result.get("explanation", ""),
            addresses_question=result.get("addresses_question", False),
            completeness_score=result.get("completeness_score", 1)
        )

    return AnswerRelevanceGrade(
        explanation="Evaluation failed",
        addresses_question=False,
        completeness_score=1
    )


def evaluate_context_precision(question: str, documents: List[str]) -> ContextPrecisionGrade:
    """Evaluate the precision of retrieved documents"""

    doc_string = "\n\n".join([f"Document {i + 1}: {doc}" for i, doc in enumerate(documents)])

    user_content = f"""QUESTION: {question}

DOCUMENTS:
{doc_string}

Evaluate which documents are relevant to answering the question."""

    result = call_groq_for_eval(CONTEXT_PRECISION_PROMPT, user_content)

    if result:
        return ContextPrecisionGrade(
            explanation=result.get("explanation", ""),
            precision_score=result.get("precision_score", 0.0),
            num_relevant=result.get("num_relevant", 0),
            num_total=result.get("num_total", len(documents))
        )

    return ContextPrecisionGrade(
        explanation="Evaluation failed",
        precision_score=0.0,
        num_relevant=0,
        num_total=len(documents)
    )


# ============================================================================
# COMPREHENSIVE RAG EVALUATION
# ============================================================================

class RAGEvaluator:
    """Comprehensive RAG system evaluator"""

    def __init__(self, chromadb_config: dict, groq_api_key: str):
        self.groq_client = Groq(api_key=groq_api_key)

        # Connect to ChromaDB
        self.client = chromadb.CloudClient(
            api_key=chromadb_config['api_key'],
            tenant=chromadb_config['tenant'],
            database=chromadb_config['database'],
        )
        self.collection = self.client.get_collection(chromadb_config['collection'])

    def rag_pipeline(self, question: str, k: int = 8) -> dict:
        """Run the RAG pipeline and return question, documents, and answer"""

        # 1. Retrieve documents
        results = self.collection.query(
            query_texts=[question],
            n_results=k,
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        # 2. Build context
        context_parts = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas)):
            place = meta.get("places", meta.get("place", "Unknown"))
            city = meta.get("city", "N/A")
            location = meta.get("location", "N/A")
            theme = meta.get("theme", "N/A")
            sentiment = meta.get("sentiment", "N/A")

            context_parts.append(
                f"Place: {place} | City: {city} | Location: {location} | "
                f"Theme: {theme} | Sentiment: {sentiment} | Notes: {doc[:200]}"
            )

        context = "\n\n".join(context_parts)

        # 3. Generate answer
        system_msg = (
            "You are a helpful travel assistant. Recommend places based ONLY on "
            "the provided context. Do not make up places or information."
        )

        user_msg = f"Question: {question}\n\nContext:\n{context}\n\nProvide recommendations:"

        completion = self.groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.3
        )

        answer = completion.choices[0].message.content

        return {
            "question": question,
            "documents": documents,
            "context": context_parts,
            "answer": answer,
            "metadatas": metadatas
        }

    def evaluate_full(self, question: str, k: int = 8) -> dict:
        """Run RAG pipeline and evaluate all metrics"""

        print(f"\n{'=' * 80}")
        print(f"EVALUATING: {question}")
        print(f"{'=' * 80}\n")

        # Run RAG
        rag_output = self.rag_pipeline(question, k)

        documents = rag_output["documents"]
        answer = rag_output["answer"]

        print(f"📥 Retrieved {len(documents)} documents")
        print(f"💬 Generated answer ({len(answer)} chars)\n")

        # Evaluate
        print("Running evaluations...")

        retrieval_grade = evaluate_retrieval_relevance(question, documents)
        print(f"✅ Retrieval Relevance: {retrieval_grade['relevant']} (score: {retrieval_grade['relevance_score']}/5)")

        answer_quality = evaluate_answer_quality(question, documents, answer)
        print(f"✅ Answer Quality: {answer_quality['quality_score']}/5")

        answer_relevance = evaluate_answer_relevance(question, answer)
        print(f"✅ Answer Relevance: {answer_relevance['completeness_score']}/5")

        context_precision = evaluate_context_precision(question, documents)
        print(f"✅ Context Precision: {context_precision['precision_score']:.2%}")

        # Compile results
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "num_documents_retrieved": len(documents),
            "answer_length": len(answer),

            "retrieval_relevance": retrieval_grade,
            "answer_quality": answer_quality,
            "answer_relevance": answer_relevance,
            "context_precision": context_precision,

            "rag_output": rag_output
        }

        return evaluation_results

    def batch_evaluate(self, test_questions: List[str], k: int = 8) -> List[dict]:
        """Evaluate multiple questions and return aggregated results"""

        results = []

        for i, question in enumerate(test_questions, 1):
            print(f"\n{'#' * 80}")
            print(f"# Test {i}/{len(test_questions)}")
            print(f"{'#' * 80}")

            result = self.evaluate_full(question, k)
            results.append(result)

        # Aggregate statistics
        print(f"\n{'=' * 80}")
        print("AGGREGATE RESULTS")
        print(f"{'=' * 80}\n")

        avg_retrieval_score = sum(r['retrieval_relevance']['relevance_score'] for r in results) / len(results)
        avg_quality_score = sum(r['answer_quality']['quality_score'] for r in results) / len(results)
        avg_relevance_score = sum(r['answer_relevance']['completeness_score'] for r in results) / len(results)
        avg_precision = sum(r['context_precision']['precision_score'] for r in results) / len(results)

        hallucination_rate = sum(1 for r in results if r['answer_quality']['hallucination_detected']) / len(results)

        print(f"📊 Average Retrieval Relevance: {avg_retrieval_score:.2f}/5")
        print(f"📊 Average Answer Quality: {avg_quality_score:.2f}/5")
        print(f"📊 Average Answer Relevance: {avg_relevance_score:.2f}/5")
        print(f"📊 Average Context Precision: {avg_precision:.2%}")
        print(f"⚠️  Hallucination Rate: {hallucination_rate:.2%}")

        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    chromadb_config = {
        'api_key': "ck-Bz6vWE8LH4mrmKx2dfreGSzgiikFBKhm7BHNHivkDvvw",
        'tenant': "6db89e03-6466-4af4-ad1c-d8237a75efa7",
        'database': "places",
        'collection': "cleaned_posts"
    }

    # Initialize evaluator
    evaluator = RAGEvaluator(chromadb_config, GROQ_API_KEY)

    # Test questions
    test_questions = [
        "What are the best pizza places in New York?",
        "Hidden gems for coffee in Seattle",
        "Best BBQ restaurants in Austin",
        "Where can I find good tacos in Los Angeles?",
        "Romantic dinner spots in San Francisco"
    ]

    # Run batch evaluation
    results = evaluator.batch_evaluate(test_questions, k=8)

    # Save results
    output_file = f"rag_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")