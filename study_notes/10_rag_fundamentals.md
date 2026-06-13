# 10 — RAG Fundamentals

## What is it?

**Retrieval-Augmented Generation (RAG)** is a technique that gives an LLM access to external knowledge. Instead of relying only on what the LLM memorized during training, RAG retrieves relevant documents from a knowledge base and feeds them into the prompt at query time.

```
User: "What is the leave policy in our company?"

Without RAG:
  LLM hallucinates a generic answer (or says "I don't know")

With RAG:
  1. Retrieve relevant pages from the company HR PDF
  2. Add those pages to the prompt
  3. LLM answers based on the retrieved text
```

---

## Why does it matter?

LLMs have two big limitations that RAG solves:

1. **Knowledge cutoff** — They don't know about recent events or private/internal data.
2. **Hallucinations** — They make up plausible-sounding but wrong answers.

RAG lets you ground the LLM in real, up-to-date, domain-specific information. It's the foundation of enterprise chatbots, document Q&A systems, legal research assistants, and internal knowledge-base search.

---

## How does it work? (Under the Hood)

### The RAG Pipeline — 6 Steps

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  1. LOAD    │────▶│ 2. CHUNK    │────▶│ 3. EMBED    │
│  documents  │     │  documents  │     │  chunks     │
│  (PDF, web, │     │  (split into│     │  (turn text │
│   text,     │     │   pieces)   │     │   into      │
│  Wikipedia) │     │             │     │   vectors)  │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 6. GENERATE │◀────│ 5. RETRIEVE │◀────│ 4. STORE    │
│  answer     │     │  relevant   │     │  vectors    │
│  using      │     │  chunks     │     │  in vector  │
│  context    │     │             │     │  database   │
└─────────────┘     └─────────────┘     └─────────────┘
```

**Step 1: Load** — Read documents from files, URLs, or APIs.

**Step 2: Chunk** — Split long documents into smaller pieces that fit in the LLM's context window.

**Step 3: Embed** — Convert each chunk into a numerical vector (embedding) that captures its meaning.

**Step 4: Store** — Save the vectors in a vector database (like FAISS).

**Step 5: Retrieve** — When a user asks a question, find the most similar chunks.

**Step 6: Generate** — Pass the retrieved chunks + user question to the LLM and generate an answer.

---

## Step 1: Loading Documents (Data Ingestion)

LangChain provides loaders for many data sources. From `7_RAG/2.1_Data_Ingestion/dataingestion.ipynb`:

### Text files

```python
from langchain_community.document_loaders.text import TextLoader

loader = TextLoader("speech.txt")
text_documents = loader.load()
print(text_documents[0].page_content)
```

### PDF files

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("rag_sample_document.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} pages")
print(docs[0].page_content[:500])
```

### Web pages

```python
from langchain_community.document_loaders import WebBaseLoader
import bs4

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        class_=("post-title", "post-content", "post-header")
    ))
)
doc = loader.load()
```

### Wikipedia

```python
from langchain_community.document_loaders import WikipediaLoader

loader = WikipediaLoader(query="Generative AI", load_max_docs=4)
docs = loader.load()
```

**Every loader returns `Document` objects** with:
- `page_content` — the actual text
- `metadata` — source, page number, etc.

---

## Step 2: Chunking Documents

Most documents are too long to feed into an LLM in one piece. Chunking breaks them into smaller, semantically meaningful pieces.

### Why chunking matters

- **Context window limits** — LLMs can only process a fixed number of tokens.
- **Better retrieval** — Smaller chunks are more focused and easier to match to queries.
- **Cost control** — You send only relevant chunks, not whole documents.

### `RecursiveCharacterTextSplitter`

This is the default recommended splitter. From `7_RAG/2.2_Data_Transformer/TextSplitter.ipynb`:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Each chunk is ~1000 characters
    chunk_overlap=200   # Overlap between chunks to preserve context
)

final_documents = text_splitter.split_documents(docs)
print(f"Number of chunks: {len(final_documents)}")
print(final_documents[0].page_content)
```

How it splits:
1. First tries to split on `\n\n` (paragraphs)
2. Then on `\n` (lines)
3. Then on `" "` (spaces)
4. Then on `""` (characters)

This keeps paragraphs and sentences together as long as possible.

### Key parameters

- **`chunk_size`** — Maximum size of each chunk (in characters or tokens, depending on splitter)
- **`chunk_overlap`** — Number of characters shared between consecutive chunks so no context is lost at boundaries

### Other splitters

```python
from langchain_text_splitters import CharacterTextSplitter, HTMLHeaderTextSplitter

# Split by a custom separator
splitter = CharacterTextSplitter(separator="\n", chunk_size=25, chunk_overlap=20)

# Split HTML by headers
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")])
```

---

## Step 3: Embeddings — Turning Text into Vectors

An **embedding** is a numerical representation of text's meaning. Similar sentences get similar vectors.

```
"King"    → [0.12, -0.34, 0.89, ..., 0.05]   (384 numbers)
"Queen"   → [0.15, -0.30, 0.85, ..., 0.08]
"Apple"   → [-0.70, 0.22, 0.10, ..., 0.44]

"King" and "Queen" are close in vector space.
"Apple" is far away.
```

### Embedding models used in the repo

**Local model (no API key):**
```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```
- Returns 384-dimensional vectors
- Runs on your machine
- Free

**OpenAI model:**
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
```
- Returns 1536-dimensional vectors
- Requires API key
- Better quality for many tasks

### Embedding a query vs. documents

```python
# Embed one query
query_embedding = embeddings.embed_query("What is RAG?")

# Embed many documents
document_embeddings = embeddings.embed_documents(list_of_text_chunks)
```

---

## Step 4 + 5: Store and Retrieve with FAISS

FAISS (Facebook AI Similarity Search) is a fast, in-memory vector database. See `11_vector_databases_and_faiss.md` for the full deep-dive.

Quick example:

```python
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store chunks
vector_store = FAISS.from_documents(final_documents, embeddings)

# Retrieve relevant chunks for a query
results = vector_store.similarity_search("What is RAG?", k=3)
for doc in results:
    print(doc.page_content)
```

---

## Step 6: Generate the Final Answer

After retrieving the top-k chunks, add them to the prompt:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer using only the provided context."),
    ("human", """Context:\n{context}\n\nQuestion: {question}""")
])

# Build context from retrieved chunks
context = "\n\n".join(doc.page_content for doc in results)

chain = prompt | llm
answer = chain.invoke({"context": context, "question": "What is RAG?"})
print(answer.content)
```

This is the complete RAG loop: **retrieve relevant context, then generate an answer grounded in that context.**

---

## How does it connect to other topics?

- **See: `01_genai_and_llm_fundamentals.md`** — RAG uses LLMs for the final generation step. Understanding context windows and token limits explains why we chunk documents.
- **See: `03_prompt_engineering.md`** — The final RAG prompt is carefully engineered: system message + context + question. Good prompting reduces hallucinations even further.
- **See: `04_pydantic_and_structured_output.md`** — You can use Pydantic to force structured answers from the RAG pipeline (e.g., citations, confidence scores).
- **See: `11_vector_databases_and_faiss.md`** — Vector databases are the storage and retrieval engine behind RAG.

---

## Code Examples

### Example 1: Minimal RAG pipeline

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. Load
loader = TextLoader("speech.txt")
docs = loader.load()

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Embed + Store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)

# 4. Retrieve
question = "What is the main message?"
retrieved = vector_store.similarity_search(question, k=3)

# 5. Generate
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the context below."),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])
chain = prompt | llm
context = "\n\n".join(d.page_content for d in retrieved)
answer = chain.invoke({"context": context, "question": question})
print(answer.content)
```

### Example 2: PDF Q&A

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("rag_sample_document.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)

vector_store = FAISS.from_documents(chunks, embeddings)
results = vector_store.similarity_search("What does this document say about AI?", k=3)
```

### Example 3: Native FAISS retriever (from `7_RAG/1_RAG_Fundamentals/RAG_Fundamentals_01.ipynb`)

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Sentences to index
sentences = ["RAG improves LLM answers", "Embeddings capture meaning", "FAISS is a vector DB"]

# Create embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(sentences)

# Build FAISS index
dimension = vectors.shape[1]  # 384
index = faiss.IndexFlatL2(dimension)
index.add(vectors.astype('float32'))

# Search
query = "How do we make LLMs more accurate?"
query_vector = model.encode([query]).astype('float32')
distances, indices = index.search(query_vector, k=2)

for idx in indices[0]:
    print(sentences[idx])
```

---

## Common Mistakes

1. **Chunks too large** — Big chunks dilute relevance and waste tokens. Start with 500-1000 characters and tune.

2. **No chunk overlap** — Without overlap, sentences split across chunk boundaries lose meaning. Use 10-20% overlap.

3. **Using the wrong embedding model** — A model trained on English won't work well for Hindi/Tamil. Match the model to your language and domain.

4. **Forgetting metadata** — Metadata like page number, source URL, and title help with debugging and citations.

5. **Retrieving too few or too many chunks** — `k=1` often misses context; `k=20` adds noise. Test with `k=3-5`.

6. **Not grounding the prompt** — If you don't explicitly instruct the LLM to use the retrieved context, it may ignore it or hallucinate.

7. **Loading everything into one big prompt** — RAG exists to avoid this. Retrieve first, then generate.

8. **Ignoring data ingestion errors** — PDFs with images, scanned pages, or weird formatting may return empty text. Always check `page_content`.

---

## Practice Exercises

1. **Build a Wikipedia Q&A bot** — Use `WikipediaLoader` to load 3 articles on a topic, chunk them, store in FAISS, and answer questions.

2. **Experiment with chunk size** — Take the same PDF and split it with `chunk_size=200`, `1000`, and `2000`. Which size gives the best answers for your test questions?

3. **Compare embedding models** — Index the same chunks with `all-MiniLM-L6-v2` and OpenAI's `text-embedding-3-small`. Run the same queries and compare relevance.

4. **Add source citations** — Modify the generation prompt to require the LLM to cite which chunk it used for each claim.

5. **Build a web-page Q&A** — Use `WebBaseLoader` to load a blog post, chunk it, and answer questions about it. Try the Lilian Weng agent post used in the repo.
