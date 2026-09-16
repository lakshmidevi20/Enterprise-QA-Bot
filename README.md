# Enterprise RAG Q&A — Pinecone + LangChain

A complete Python Retrieval-Augmented Generation (RAG) project for enterprise documents such as:

- HR policies
- compliance/security manuals
- product documentation
- onboarding guides

It ingests local PDF, DOCX, TXT, and Markdown files, chunks them, creates embeddings, stores vectors in Pinecone, retrieves relevant context, generates grounded answers with LangChain, shows sources, and runs a 15-question stress test.

## Architecture

```text
Enterprise Documents
      |
      v
PDF / DOCX / TXT / MD loaders
      |
      v
RecursiveCharacterTextSplitter
      |
      v
OpenAI embeddings
      |
      v
Pinecone vector index
      |
User Question
      |
      v
Top-k semantic retrieval
      |
      +--> relevance gate --> "I don't have enough information..."
      |
      v
Grounded prompt + ChatOpenAI
      |
      v
Answer + source citations
```

## Project structure

```text
enterprise_rag_qa/
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── hr_policy.md
│   ├── security_compliance.md
│   ├── product_guide.md
│   └── onboarding_guide.md
├── reports/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── loaders.py
│   ├── vectorstore.py
│   ├── ingest.py
│   └── rag.py
└── tests/
    ├── stress_questions.json
    └── run_stress_test.py
```

## 1. Prerequisites

- Python 3.11+ recommended
- Pinecone account/API key
- OpenAI API key with API billing/credits enabled

## 2. Create and activate a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Copy:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Then edit `.env`:

```dotenv
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=enterprise-rag-demo
```

Do not commit `.env`.

## 5. Add documents

The repository includes four small demo enterprise documents.

To use real documents, put PDF, DOCX, TXT, or Markdown files into:

```text
data/
```

Subfolders are supported.

For scanned/image-only PDFs, this basic loader is not enough because `pypdf` extracts embedded text rather than performing OCR. Use an OCR/document-intelligence loader for those files.

## 6. Ingest and index

From the project root:

```bash
python -m src.ingest --data-dir data
```

The script will:

1. load supported documents;
2. retain source and page metadata where available;
3. split documents into overlapping chunks;
4. create OpenAI embeddings;
5. create the Pinecone serverless index if needed;
6. upsert chunks using stable IDs.

Stable IDs mean rerunning ingestion replaces matching vectors rather than endlessly duplicating identical chunks.

## 7. Start the Q&A interface

```bash
streamlit run app.py
```

Then ask questions such as:

```text
How many paid vacation days do full-time employees receive?
```

or:

```text
A new engineer needs access to production customer data.
What onboarding and security steps apply?
```

## 8. Run the 15-question stress test

First ingest the sample documents, then run:

```bash
python tests/run_stress_test.py
```

It tests:

- direct answerable questions;
- multi-document questions;
- ambiguous questions;
- unanswerable questions;
- a cross-document policy consistency case.

Outputs:

```text
reports/stress_test_report.json
reports/stress_test_report.csv
```

## What the test measures

Each question checks:

- whether the RAG system answered/refused appropriately;
- retrieval relevance score;
- sources retrieved;
- whether expected source documents were retrieved;
- overall pass/fail.

This is deliberately a lightweight evaluation harness. In a production RAG system you would also measure:

- retrieval recall@k;
- precision@k;
- answer faithfulness;
- groundedness;
- context relevance;
- answer correctness against human-labeled reference answers;
- latency and token cost.

## Edge-case handling

### Unanswerable questions

The system has two protections:

1. a retrieval relevance threshold (`MIN_RELEVANCE_SCORE`);
2. a strict prompt telling the LLM to answer only from context.

If retrieval is too weak, the LLM is bypassed completely and the system returns:

```text
I don't have enough information in the knowledge base to answer that.
```

### Ambiguous questions

The prompt tells the model to state the most likely interpretation and ask for clarification instead of inventing missing context.

### Questions spanning multiple documents

`TOP_K=5` allows chunks from several source documents to enter the prompt. Answers are instructed to combine them and cite each relevant source.

## Important tuning knobs

In `.env`:

```dotenv
CHUNK_SIZE=900
CHUNK_OVERLAP=150
TOP_K=5
MIN_RELEVANCE_SCORE=0.40
```

Do not treat `0.40` as a universal best threshold. Tune it against your own labeled question set because score distributions vary with content, embedding model, and retrieval setup.

## Suggested production improvements

1. Add metadata such as department, policy version, effective date, region, security classification, and ACL.
2. Enforce access control during retrieval so a user never receives chunks they are not authorized to see.
3. Add document versioning and deletion handling.
4. Use hybrid dense + keyword search where exact policy IDs, product codes, or acronyms matter.
5. Add reranking after Pinecone retrieval.
6. Replace simple file parsing with a layout-aware parser for tables and complex PDFs.
7. Add LangSmith or another observability/evaluation layer.
8. Maintain a human-reviewed golden evaluation dataset.
9. Add PII/secret filtering and audit logging.
10. Add incremental ingestion based on file hashes/version IDs.

## Troubleshooting

### OpenAI 429 / insufficient quota

If you receive an error such as:

```text
429 insufficient_quota
```

your OpenAI API account does not currently have usable API credit/billing. ChatGPT subscription billing and OpenAI API billing are separate.

### `NameError: response is not defined`

This usually happens when an earlier API call failed before assigning a `response` variable. Fix the original API error first and only reference the variable after a successful call.

### Wrong Pinecone dimension

The Pinecone index dimension must match your embedding output. This starter uses:

```text
text-embedding-3-small + dimensions=1536
```

If you change the embedding dimension, create a new index or recreate the old one with the matching dimension.

## Security warning

Do not put secrets, API keys, passwords, or restricted documents into a demo environment without appropriate enterprise controls. A production enterprise RAG application should enforce authentication, authorization, encryption, auditing, retention, and document-level permissions.
