import os
import re
from pathlib import Path

import chromadb
import fitz # PyMuPDF
from openai import OpenAI


# Settings 


PDF_FILE = "Hyderabad_Business_Plan.pdf"
DATABASE_FOLDER = "chroma_pdf_reader_db"
COLLECTION_NAME = "laundry_business_chunks"
OPENAI_MODEL = "gpt-4o-mini"

CHUNK_SIZE = 150
CHUNK_OVERLAP = 25
FIRST_SEARCH_K = 8
FINAL_RESULT_K = 3

# ---------------------------------------------------------------------------
# Step 1: Read and clean the PDF
# ---------------------------------------------------------------------------

def read_pdf(pdf_file):
    pdf_path = Path(pdf_file)
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Could not find {pdf_path}. Make sure PDF is in same folder as main.py"
        )
    pages = []
    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            pages.append({
                "page_number": page_number,
                "text": page.get_text("text"),
            })
    return pages

def clean_text(text):
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

#  Chunking


def make_chunks(pages):
    chunks = []
    chunk_number = 1
    for page in pages:
        words = clean_text(page["text"]).split()
        start = 0
        while start < len(words):
            end = start + CHUNK_SIZE
            chunk_text = " ".join(words[start:end])
            chunks.append({
                "id": f"chunk-{chunk_number:04d}",
                "text": chunk_text,
                "page_number": page["page_number"],
            })
            chunk_number += 1
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

#  Embedding and indexing with local Chroma


def build_index(chunks):
    client = chromadb.PersistentClient(path=DATABASE_FOLDER)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)
    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {"page_number": chunk["page_number"], "source": PDF_FILE}
            for chunk in chunks
        ],
    )
    return collection

def show_embedding_example(collection, first_chunk_id):
    saved_item = collection.get(
        ids=[first_chunk_id],
        include=["documents", "embeddings"],
    )
    embeddings = saved_item.get("embeddings")
    if embeddings is None or len(embeddings) == 0:
        print("Embedding could not be displayed.")
        return
    vector = embeddings[0]
    print("\nEmbedding example")
    print("=" * 70)
    print("An embedding is a long list of numbers representing text meaning.")
    print(f"Vector length: {len(vector)} numbers")
    print(f"First 8 numbers: {[round(float(number), 4) for number in vector[:8]]}")

# ---------------------------------------------------------------------------
# Step 4: Similarity search and retrieval
# ---------------------------------------------------------------------------

def retrieve(collection, search_text, top_k):
    raw = collection.query(
        query_texts=[search_text],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for chunk_id, text, metadata, distance in zip(
        raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0],
    ):
        results.append({
            "id": chunk_id,
            "text": text,
            "page_number": metadata["page_number"],
            "distance": float(distance),
        })
    return results

def print_results(title, results):
    print(f"\n{title}")
    print("=" * 70)
    for position, result in enumerate(results, start=1):
        rerank_text = ""
        if "rerank_score" in result:
            rerank_text = f", rerank_score={result['rerank_score']:.3f}"
        print(f"{position}. page={result['page_number']}, distance={result['distance']:.3f}{rerank_text}")
        print(f" {result['text'][:280]}\n")

# ---------------------------------------------------------------------------
# Step 5: Simple reranking
# ---------------------------------------------------------------------------

def words_in(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

def rerank(question, results, keep):
    question_words = words_in(question)
    for result in results:
        chunk_words = words_in(result["text"])
        vector_score = 1 / (1 + result["distance"])
        if question_words:
            keyword_score = len(question_words & chunk_words) / len(question_words)
        else:
            keyword_score = 0
        result["rerank_score"] = (0.8 * vector_score) + (0.2 * keyword_score)
    results.sort(key=lambda item: item["rerank_score"], reverse=True)
    return results[:keep]

# ---------------------------------------------------------------------------
# OpenAI helper
# ---------------------------------------------------------------------------

def ask_openai(instructions, user_input):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError('OPENAI_API_KEY is not set. Run: $env:OPENAI_API_KEY="your-key"')
    client = OpenAI()
    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=user_input,
    )
    return response.output_text.strip()

# ---------------------------------------------------------------------------
# Step 6: Query decomposition
# ---------------------------------------------------------------------------

def decompose_question(question):
    text = ask_openai(
        instructions="Split the user's question into at most three short search questions. Return one question per line. Do not add explanations or numbering.",
        user_input=question,
    )
    questions = [line.strip(" -1234567890.\t") for line in text.splitlines()]
    return [item for item in questions if item][:3]

def retrieve_with_decomposition(collection, original_question):
    smaller_questions = decompose_question(original_question)
    combined = {}
    print("\nDecomposed questions")
    print("=" * 70)
    for question in smaller_questions:
        print(f"- {question}")
        for result in retrieve(collection, question, top_k=3):
            old_result = combined.get(result["id"])
            if old_result is None or result["distance"] < old_result["distance"]:
                combined[result["id"]] = result
    return list(combined.values())

# ---------------------------------------------------------------------------
# Step 7: HyDE
# ---------------------------------------------------------------------------

def create_hypothetical_answer(question):
    return ask_openai(
        instructions="Write a short two-sentence passage that could answer the question. It is only a search aid, so do not claim that it is factually correct.",
        user_input=question,
    )

def retrieve_with_hyde(collection, question):
    hypothetical_answer = create_hypothetical_answer(question)
    print("\nHyDE hypothetical answer used for search")
    print("=" * 70)
    print(hypothetical_answer)
    return retrieve(collection, hypothetical_answer, top_k=3)

# ---------------------------------------------------------------------------
# Step 8: Grounded answer
# ---------------------------------------------------------------------------

def answer_with_sources(question, results):
    source_blocks = []
    for number, result in enumerate(results, start=1):
        source_blocks.append(f"SOURCE {number} - page {result['page_number']}\n{result['text']}")
    context = "\n\n".join(source_blocks)
    return ask_openai(
        instructions="Answer only from the supplied PDF sources. Keep the answer short and clear for a student. Cite page numbers like [page 4]. If the sources do not contain the answer, say: I could not find this in the PDF.",
        user_input=f"QUESTION:\n{question}\n\nPDF SOURCES:\n{context}",
    )

# ---------------------------------------------------------------------------
# Run the complete lesson
# ---------------------------------------------------------------------------

def main():
    print("STAGE 1 - Read and chunk the PDF")
    pages = read_pdf(PDF_FILE)
    chunks = make_chunks(pages)
    if not chunks:
        print("No text was found. This may be a scanned PDF that needs OCR.")
        return
    print(f"Pages read: {len(pages)}")
    print(f"Chunks created: {len(chunks)}")

    print("\nSTAGE 2 - Build the local Chroma index")
    collection = build_index(chunks)
    print(f"Items in the index: {collection.count()}")
    show_embedding_example(collection, chunks[0]["id"])

    question = input("\nAsk a question about the PDF: ").strip()
    if not question:
        print("No question was entered.")
        return

    print("\nSTAGE 3 - Basic similarity retrieval")
    basic_results = retrieve(collection, question, FIRST_SEARCH_K)
    print_results("Initial Chroma results", basic_results[:FINAL_RESULT_K])

    print("STAGE 4 - Reranking")
    reranked_results = rerank(question, basic_results, FINAL_RESULT_K)
    print_results("Results after reranking", reranked_results)

    print("STAGE 5 - Decomposition")
    decomposition_results = retrieve_with_decomposition(collection, question)

    print("\nSTAGE 6 - HyDE")
    hyde_results = retrieve_with_hyde(collection, question)

    all_candidates = {}
    for result in reranked_results + decomposition_results + hyde_results:
        old_result = all_candidates.get(result["id"])
        if old_result is None or result["distance"] < old_result["distance"]:
            all_candidates[result["id"]] = result

    final_results = rerank(question, list(all_candidates.values()), FINAL_RESULT_K)
    print_results("Final retrieved evidence", final_results)

    print("STAGE 7 - Grounded answer from gpt-4o-mini")
    answer = answer_with_sources(question, final_results)
    print("\nFinal answer")
    print("=" * 70)
    print(answer)

if __name__ == "__main__":
    main()