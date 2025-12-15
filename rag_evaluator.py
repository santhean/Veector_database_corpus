"""
Simple RAG Evaluation Module
Easy to integrate into your existing RAG application
"""

import json
from typing import List, Dict
from groq import Groq


class SimpleRAGEvaluator:
    """Lightweight RAG evaluator using Groq"""

    def __init__(self, groq_api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=groq_api_key)
        self.model = model

    def _call_llm(self, system_prompt: str, user_prompt: str) -> dict:
        """Helper to call LLM with JSON response"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")
            return {}

    def evaluate_retrieval(self, question: str, documents: List[str]) -> Dict:
        """
        Evaluate if retrieved documents are relevant to the question

        Returns:
            dict with keys: relevant (bool), score (int 1-5), explanation (str)
        """
        system_prompt = """You are an evaluator. Grade if retrieved documents are relevant to a question.

Respond in JSON with:
- relevant: true/false
- score: 1-5 (1=not relevant, 5=highly relevant)
- explanation: brief reasoning"""

        docs_text = "\n\n".join([f"Doc {i + 1}: {doc[:200]}" for i, doc in enumerate(documents)])
        user_prompt = f"Question: {question}\n\nDocuments:\n{docs_text}\n\nAre these relevant?"

        result = self._call_llm(system_prompt, user_prompt)
        return {
            'relevant': result.get('relevant', False),
            'score': result.get('score', 1),
            'explanation': result.get('explanation', 'N/A')
        }

    def evaluate_answer(self, question: str, documents: List[str], answer: str) -> Dict:
        """
        Evaluate the quality of generated answer

        Returns:
            dict with keys: is_grounded (bool), is_helpful (bool),
                          has_hallucination (bool), quality_score (int 1-5),
                          explanation (str)
        """
        system_prompt = """You are an evaluator. Grade an AI-generated answer.

Check:
1. is_grounded: based on provided facts (not made up)
2. is_helpful: actually helps the user
3. has_hallucination: mentions things NOT in the facts
4. quality_score: 1-5 overall quality

Respond in JSON with those keys plus explanation."""

        docs_text = "\n\n".join([f"Fact {i + 1}: {doc[:150]}" for i, doc in enumerate(documents)])
        user_prompt = f"Question: {question}\n\nFacts:\n{docs_text}\n\nAnswer: {answer}\n\nEvaluate:"

        result = self._call_llm(system_prompt, user_prompt)
        return {
            'is_grounded': result.get('is_grounded', False),
            'is_helpful': result.get('is_helpful', False),
            'has_hallucination': result.get('has_hallucination', True),
            'quality_score': result.get('quality_score', 1),
            'explanation': result.get('explanation', 'N/A')
        }

    def evaluate_relevance(self, question: str, answer: str) -> Dict:
        """
        Evaluate if answer addresses the question

        Returns:
            dict with keys: addresses_question (bool), completeness (int 1-5),
                          explanation (str)
        """
        system_prompt = """You are an evaluator. Check if an answer addresses the question.

Respond in JSON with:
- addresses_question: true/false
- completeness: 1-5 (how completely it answers)
- explanation: brief reasoning"""

        user_prompt = f"Question: {question}\n\nAnswer: {answer}\n\nDoes it address the question?"

        result = self._call_llm(system_prompt, user_prompt)
        return {
            'addresses_question': result.get('addresses_question', False),
            'completeness': result.get('completeness', 1),
            'explanation': result.get('explanation', 'N/A')
        }

    def evaluate_all(self, question: str, documents: List[str], answer: str) -> Dict:
        """Run all evaluations and return combined results"""

        print(f"\nEvaluating: '{question[:50]}...'")

        retrieval = self.evaluate_retrieval(question, documents)
        status = "PASS" if retrieval['relevant'] else "FAIL"
        print(f"  Retrieval: {status} (score: {retrieval['score']}/5)")

        answer_quality = self.evaluate_answer(question, documents, answer)
        print(f"  Answer Quality: {answer_quality['quality_score']}/5")

        relevance = self.evaluate_relevance(question, answer)
        print(f"  Relevance: {relevance['completeness']}/5")

        return {
            'question': question,
            'retrieval_evaluation': retrieval,
            'answer_quality': answer_quality,
            'answer_relevance': relevance,
            'overall_score': (
                                     retrieval['score'] +
                                     answer_quality['quality_score'] +
                                     relevance['completeness']
                             ) / 3
        }


def example_usage():
    """Example of how to use the evaluator"""

    # Initialize
    evaluator = SimpleRAGEvaluator(
        groq_api_key="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"
    )

    # Example data
    question = "What are good pizza places in New York?"

    documents = [
        "Joe's Pizza in Greenwich Village is a classic NYC spot known for its thin crust slices.",
        "Di Fara Pizza in Brooklyn is considered one of the best pizza places in the city.",
        "Central Park is a large public park in Manhattan with many walking trails."
    ]

    answer = "Based on the information, I recommend Joe's Pizza in Greenwich Village and Di Fara Pizza in Brooklyn. Both are highly regarded for their excellent pizza."

    # Evaluate
    results = evaluator.evaluate_all(question, documents, answer)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print(f"\nOverall Score: {results['overall_score']:.2f}/5")


def integrate_with_your_rag(rag_search_and_answer_function):
    """
    Example of how to integrate evaluation into your existing RAG function

    Usage:
        # In your existing code, after generating an answer:
        evaluator = SimpleRAGEvaluator(groq_api_key=GROQ_API_KEY)

        eval_results = evaluator.evaluate_all(
            question=query,
            documents=documents,
            answer=answer_text
        )

        # Now you have evaluation metrics
    """
    pass


if __name__ == "__main__":
    example_usage()