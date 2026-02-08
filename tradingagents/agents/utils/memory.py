import os

import chromadb
from chromadb.config import Settings
from openai import OpenAI

# Default OpenAI embeddings endpoint
_OPENAI_BASE_URL = "https://api.openai.com/v1"


class FinancialSituationMemory:
    def __init__(self, name, config):
        # Determine embedding model: explicit config > auto-detect by provider
        if config.get("embedding_model"):
            self.embedding = config["embedding_model"]
        elif config["backend_url"] == "http://localhost:11434/v1":
            self.embedding = "nomic-embed-text"
        else:
            self.embedding = "text-embedding-3-small"

        # Embedding endpoint: separate config > LLM backend (only when provider
        # is standard OpenAI) > OpenAI default.
        #
        # When the user configures a custom/non-OpenAI LLM provider (e.g.
        # DeepSeek, Groq) but does NOT set explicit embedding config, we must
        # NOT send the OpenAI embedding request to the custom provider — it
        # won't serve text-embedding-3-small.  Fall back to the OpenAI API
        # using OPENAI_API_KEY from the environment instead.
        embedding_base_url = config.get("embedding_base_url")
        embedding_api_key = config.get("embedding_api_key")

        if not embedding_base_url:
            provider = config.get("llm_provider", "openai").lower()
            if provider == "openai":
                # Standard OpenAI — safe to reuse the LLM backend/key
                embedding_base_url = config["backend_url"]
                embedding_api_key = embedding_api_key or config.get("api_key")
            else:
                # Custom / non-OpenAI provider — fall back to OpenAI for embeddings
                embedding_base_url = _OPENAI_BASE_URL
                embedding_api_key = embedding_api_key or os.getenv("OPENAI_API_KEY")

        client_kwargs = {"base_url": embedding_base_url}
        if embedding_api_key:
            client_kwargs["api_key"] = embedding_api_key
        self.client = OpenAI(**client_kwargs)

        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""
        
        response = self.client.embeddings.create(
            model=self.embedding, input=text
        )
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
