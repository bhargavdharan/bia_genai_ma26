# 11 — Vector Databases & FAISS

## What is it?

A **vector database** stores data as high-dimensional vectors (embeddings) instead of rows and columns. It lets you search by **semantic similarity** — "find me things that mean similar things" — rather than by exact keyword matches.

```
Traditional Database Query:
  "SELECT * FROM products WHERE name LIKE '%laptop%'"
  → Only finds products with "laptop" in the name

Vector Database Query:
  "Find vectors close to the meaning of 'portable computer for coding'"
  → Finds laptops, ultrabooks, notebooks, MacBooks — even if they don't contain the exact words
```

---

## Why does it matter?

Vector databases power:
- **Semantic search** — find conceptually related content
- **RAG systems** — retrieve relevant documents for LLMs
- **Recommendation engines** — "users who liked X also liked Y"
- **Image/Audio search** — search by content, not metadata

Without vector databases, RAG would have to compare every query against every chunk using slow, brute-force methods. Vector DBs make similarity search fast even with millions of chunks.

---

## How does it work? (Under the Hood)

### From Text to Vector

```
"The cat sat on the mat"
        │
        ▼
   Embedding Model
        │
        ▼
[0.12, -0.45, 0.89, ..., 0.03]   ← a vector of numbers
```

The embedding model converts text into a fixed-length vector (e.g., 384 or 1536 dimensions). The model is trained so that **semantically similar texts produce vectors that are close together** in this high-dimensional space.

### Visualizing Similarity (2D simplification)

```
                    King
                     │
        Man ◀───────┼───────▶ Woman
                     │
                    Queen

"King - Man + Woman ≈ Queen"
```

In the real vector space, you can't visualize 384 dimensions, but the math works the same way:
- Similar meanings → small distance / high cosine similarity
- Different meanings → large distance / low cosine similarity

### Similarity Metrics

#### Cosine Similarity

Measures the **angle** between two vectors, ignoring their magnitude.

```python
from sklearn.metrics.pairwise import cosine_similarity

# documents: list of vectors
# query: single vector (wrapped in a list)
scores = cosine_similarity(document_embeddings, [query_embedding])
```

- Range: `-1` to `1`
- `1` = identical direction (very similar)
- `0` = orthogonal (unrelated)
- Most common for text embeddings

#### Euclidean (L2) Distance

Measures the **straight-line distance** between two vector points.

```python
from sklearn.metrics.pairwise import euclidean_distances

distances = euclidean_distances(document_embeddings, [query_embedding])
```

- Range: `0` to infinity
- `0` = identical
- Smaller = more similar
- FAISS `IndexFlatL2` uses this metric

From `7_RAG/3_FAISS/vectordbnotes.ipynb`:

```python
documents = [
    "What is the capital of germany?",
    "Who is the president of germany?",
    "What is the population of germany?"
]

my_query = "Narendra Modi is the prime minister of which country?"

document_embeddings = embeddings.embed_documents(documents)
query_embedding = embeddings.embed_query(my_query)

# Cosine similarity
cosine_similarity(document_embeddings, [query_embedding])

# Euclidean distance
euclidean_distances(document_embeddings, [query_embedding])
```

---

## FAISS — Facebook AI Similarity Search

FAISS is an open-source library for efficient similarity search. It's the vector database used throughout the RAG notebooks in this repo.

### Native FAISS (low-level)

From `7_RAG/1_RAG_Fundamentals/RAG_Fundamentals_01.ipynb`:

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sentences = ["RAG improves LLM answers", "Embeddings capture meaning", "FAISS is fast"]
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(sentences)

# 1. Create index
dimension = vectors.shape[1]  # 384 for MiniLM
index = faiss.IndexFlatL2(dimension)

# 2. Add vectors (must be float32)
index.add(vectors.astype('float32'))
print(f"Indexed {index.ntotal} vectors")

# 3. Search
query = "How do we make LLMs more accurate?"
query_vector = model.encode([query]).astype('float32')
distances, indices = index.search(query_vector, k=2)

for idx in indices[0]:
    print(sentences[idx])
```

**Key objects:**
- `IndexFlatL2` — exact search using L2 distance. Simple, always correct, but slower for huge datasets.
- `index.add()` — inserts vectors
- `index.search()` — returns distances and indices of nearest neighbors

### FAISS with LangChain (high-level)

From `7_RAG/3_FAISS/vectordbnotes.ipynb`:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryVectorStore

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Build the vector store
vector_store = FAISS.from_texts(
    ["Hello, my dog is cute", "AI Agents are powerful", "AI is the future"],
    embeddings
)

# Or from documents
vector_store = FAISS.from_documents(chunks, embeddings)

# Search
results = vector_store.similarity_search("Tell me about AI", k=2)
for doc in results:
    print(doc.page_content)
```

### Manual FAISS + LangChain construction

If you want full control over the index:

```python
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create FAISS index manually
index = faiss.IndexFlatL2(384)  # 384 dimensions for MiniLM

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryVectorStore(),
    index_to_docstore_id={}
)

# Add texts
vector_store.add_texts(["Hello, my dog is cute", "AI Agents are powerful"])

# Inspect mapping from FAISS index → document ID
faiss_index = 0
docstore_id = vector_store.index_to_docstore_id[faiss_index]
print(vector_store.docstore.search(docstore_id))
```

---

## Popular Vector Databases

```
Database        Type                Best For
─────────────────────────────────────────────────────────
FAISS           In-memory library   Prototyping, small/medium
Chroma          Embedded/local      Easy Python apps
Pinecone        Managed cloud       Production, scaling
Weaviate        Open-source/self    Hybrid search, GraphQL
Qdrant          Open-source         Filtering, high performance
Milvus          Distributed         Large-scale enterprise
```

For learning and small projects, **FAISS** is enough. For production with millions of vectors and persistence, you typically move to Chroma, Pinecone, Qdrant, or Weaviate.

---

## Embedding Models Comparison

```
Model                          Dimensions    Provider        Cost
───────────────────────────────────────────────────────────────────
all-MiniLM-L6-v2               384           HuggingFace     Free (local)
text-embedding-3-small         1536          OpenAI          Pay per token
text-embedding-3-large         3072          OpenAI          More expensive
```

**Rule of thumb:**
- Start with `all-MiniLM-L6-v2` for free, fast experiments.
- Upgrade to OpenAI embeddings if retrieval quality is critical and you have budget.

---

## How does it connect to other topics?

- **See: `10_rag_fundamentals.md`** — Vector databases are the storage and retrieval layer of every RAG pipeline.
- **See: `04_pydantic_and_structured_output.md`** — You can structure retrieval results with Pydantic (e.g., ranked results with scores and sources).
- **See: `01_genai_and_llm_fundamentals.md`** — Embeddings come from the same transformer technology as LLMs. Understanding tokens helps you understand why chunking matters.

---

## Code Examples

### Example 1: Compare two sentences with embeddings

```python
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

sentences = ["Machine learning is amazing", "Deep learning is a subset of ML"]
vectors = embeddings.embed_documents(sentences)

score = cosine_similarity([vectors[0]], [vectors[1]])
print(score)  # High value = similar meaning
```

### Example 2: Basic similarity search with FAISS

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

docs = [
    "The sky is blue",
    "Grass is green",
    "The sun is a star"
]

vector_store = FAISS.from_texts(docs, embeddings)
results = vector_store.similarity_search("What color is the sky?", k=2)

for r in results:
    print(r.page_content)
```

### Example 3: Native FAISS with scores

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sentences = ["Cat", "Dog", "Car", "Truck"]
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(sentences).astype('float32')

index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

query = model.encode(["Automobile"]).astype('float32')
distances, indices = index.search(query, k=2)

for dist, idx in zip(distances[0], indices[0]):
    print(f"{sentences[idx]} (distance: {dist:.4f})")
# Output likely: Car, Truck
```

### Example 4: Save and load a FAISS index

```python
# Save
vector_store.save_local("faiss_index")

# Load
from langchain_community.vectorstores import FAISS
loaded_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
```

> ⚠️ `allow_dangerous_deserialization=True` is required because FAISS indexes can contain arbitrary Python objects. Only load indexes you created yourself.

---

## Common Mistakes

1. **Mixing embedding models** — If you index with MiniLM but query with OpenAI, the vectors live in completely different spaces. Use the **same embedding model** for indexing and querying.

2. **Wrong distance metric** — Cosine similarity and L2 distance behave differently. Make sure your index matches your intended comparison. `IndexFlatIP` for inner product, `IndexFlatL2` for Euclidean.

3. **Forgetting `float32`** — FAISS requires vectors as `float32`. Passing `float64` will fail or behave unexpectedly.

4. **Not normalizing vectors for cosine** — If using `IndexFlatIP` (inner product) to approximate cosine similarity, normalize your vectors first. Or use a library that handles it for you.

5. **Storing everything in memory** — FAISS is in-memory by default. For large datasets, use disk-backed options or a managed vector DB like Pinecone/Qdrant.

6. **Ignoring dimension mismatch** — If your index is built for 384-dimensional vectors, you cannot add 1536-dimensional vectors. The dimension must match the embedding model.

7. **Retrieving without metadata** — Raw vectors don't tell you where the text came from. Always keep a mapping from vector ID back to original document/chunk.

8. **Expecting keyword matching** — Vector search is semantic. A query for "laptop" may return "MacBook" or "notebook" but miss a document that literally says "laptop" if the embedding model doesn't associate them strongly.

---

## Practice Exercises

1. **Build a tiny semantic search engine** — Create a list of 10 sentences about different animals. Embed them with MiniLM and use FAISS to find the closest sentence to queries like "fast runner" or "lives in water".

2. **Compare metrics** — Index the same vectors with both L2 distance and cosine similarity (via normalized vectors + inner product). Run the same queries and compare the rankings.

3. **Test embedding model quality** — Index the same chunks with `all-MiniLM-L6-v2` and OpenAI `text-embedding-3-small`. Query both and decide which returns more relevant results for your data.

4. **Add documents incrementally** — Create an empty FAISS index, then call `add_texts()` multiple times with new documents. Verify that `index.ntotal` grows.

5. **Build a persistent search index** — Create a FAISS index, save it to disk with `save_local()`, delete the variable, then reload it with `load_local()` and run a query. This simulates a real production deployment.
