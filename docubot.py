"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "and", "or",
    "but", "if", "so", "it", "its", "this", "that", "these", "those",
    "i", "my", "me", "we", "our", "you", "your", "what", "how", "where",
    "when", "why", "who", "which", "not", "no", "get", "make", "up",
    "than", "then", "there", "here", "just", "also", "more", "very"
}

_MIN_SCORE = 3          # top snippet must score at least this to count as evidence
_MIN_COVERAGE_RATIO = 0.25  # at least 25% of meaningful query words must exist in the index


class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        TODO (Phase 1):
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.

        Example structure:
        {
            "token": ["AUTH.md", "API_REFERENCE.md"],
            "database": ["DATABASE.md"]
        }

        Keep this simple: split on whitespace, lowercase tokens,
        ignore punctuation if needed.
        """
        index = {}
        for filename, text in documents:
            for token in text.split():
                word = token.lower().strip(".,!?;:\"'()[]{}")
                if not word:
                    continue
                if word not in index:
                    index[word] = []
                if filename not in index[word]:
                    index[word].append(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        TODO (Phase 1):
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        query_words = [w.lower().strip(".,!?;:\"'()[]{}") for w in query.split()]
        query_words = [w for w in query_words if w]

        text_words = [w.lower().strip(".,!?;:\"'()[]{}") for w in text.split()]
        text_words = [w for w in text_words if w]

        # How many times each query word appears in the text (frequency)
        frequency_score = sum(text_words.count(word) for word in query_words)

        # How many unique query words are covered (coverage)
        text_word_set = set(text_words)
        coverage_score = sum(1 for word in set(query_words) if word in text_word_set)

        return frequency_score + coverage_score

    def retrieve(self, query, top_k=3):
        """
        TODO (Phase 1):
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        results = []

        # Use the index to find candidate documents that contain query words
        query_words = [w.lower().strip(".,!?;:\"'()[]{}") for w in query.split()]
        query_words = [w for w in query_words if w]

        candidate_filenames = set()
        for word in query_words:
            if word in self.index:
                candidate_filenames.update(self.index[word])

        # Score each candidate document and add the best matching paragraph
        for filename, text in self.documents:
            if filename in candidate_filenames:
                paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
                best = max(paragraphs, key=lambda p: self.score_document(query, p))
                score = self.score_document(query, best)
                if score > 0:
                    results.append((filename, best, score))

        # Sort by score descending, then strip the score before returning
        results.sort(key=lambda x: x[2], reverse=True)
        results = [(filename, text) for filename, text, _ in results]

        return results[:top_k]

    # -----------------------------------------------------------
    # Confidence Check
    # -----------------------------------------------------------

    def _refusal_reason(self, query, results):
        """
        Returns a refusal message string if the query cannot be answered
        confidently, or None if the results look sufficient.

        Catches:
        - Too vague: no meaningful (non-stop-word) terms in the query
        - Out of scope: fewer than 25% of meaningful terms exist in the index
        - Too complex / weak match: top snippet score is below the minimum threshold
        - No evidence: retrieval returned nothing
        """
        all_words = [w.lower().strip(".,!?;:\"'()[]{}") for w in query.split()]
        meaningful = [w for w in all_words if w and w not in _STOP_WORDS]

        if not meaningful:
            return (
                "Your query is too vague. "
                "Please ask a specific question about the documentation."
            )

        covered = sum(1 for w in meaningful if w in self.index)
        coverage_ratio = covered / len(meaningful)

        if coverage_ratio < _MIN_COVERAGE_RATIO:
            return (
                "That topic does not appear to be covered in the available docs. "
                "I don't have enough relevant information to answer."
            )

        if not results:
            return "I could not find any relevant information in the docs for that query."

        top_score = self.score_document(query, results[0][1])
        if top_score < _MIN_SCORE:
            return (
                "The available documentation mentions this topic only in passing. "
                "I don't have enough context to give a reliable answer."
            )

        return None

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        refusal = self._refusal_reason(query, snippets)
        if refusal:
            return refusal

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        refusal = self._refusal_reason(query, snippets)
        if refusal:
            return refusal

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
