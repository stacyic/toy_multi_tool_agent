# Multi-Tool Agent

A multi-tool LangChain agent with RAG, Text-to-SQL, semantic evaluation, and safety features.

## Features

- **RAG**: Policy retrieval via FAISS with similarity scoring and source citations
- **Text-to-SQL**: Query generation with 3x retry loop and semantic validation
- **Semantic Evaluation**: LLM-as-judge validates SQL correctness, flags ambiguous queries
- **Human Review Queue**: Auto-captures failed/ambiguous queries for review
- **Safety**: PII masking, SQL injection detection, read-only enforcement

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env  # Add OPENAI_API_KEY
```

## Usage

```bash
# Initialize database
.venv/bin/python3 -m multi_tool_agent.main init-db --data-dir ./data

# Query with different options
.venv/bin/python3 -m multi_tool_agent.main query "What is the return policy?"
.venv/bin/python3 -m multi_tool_agent.main query --eval "How many VIP customers?"      # Semantic eval
.venv/bin/python3 -m multi_tool_agent.main query --rag-metrics "VIP benefits?"         # RAG metrics
.venv/bin/python3 -m multi_tool_agent.main query --perf "List low stock products"      # Performance

# Batch processing
.venv/bin/python3 -m multi_tool_agent.main batch --file queries.json --output results.json
```

## CLI Flags

| Flag | Description |
|------|-------------|
| `--eval`, `-e` | Enable semantic evaluation (LLM-as-judge) |
| `--rag-metrics`, `-r` | Show retrieval similarity scores and sources |
| `--perf`, `-p` | Component-level performance timing |
| `--verbose`, `-v` | Debug output |

## Architecture

```
multi_tool_agent/
├── core/
│   ├── agent.py              # Main orchestration
│   ├── query_router.py       # Tool selection
│   └── batch_processor.py    # Batch optimization
├── tools/
│   ├── policy_accessor.py    # RAG with FAISS
│   ├── rag_metrics.py        # Retrieval tracking
│   └── sql_accessor/
│       ├── sql_tool.py          # SQL with retry
│       ├── query_generator.py   # LLM SQL generation
│       ├── semantic_evaluator.py # LLM-as-judge
│       ├── human_review.py      # Review queue
│       └── pii_masker.py        # PII detection
├── stores/faiss_store.py
├── logging/
└── main.py
```

## Data Files

| File | Description |
|------|-------------|
| `data/store.db` | SQLite database |
| `data/policies.md` | Business policies for RAG |
| `data/review_queue.json` | Flagged queries for human review |
| `logs/agent.traces.jsonl` | Request traces with costs |

## Configuration

```bash
# .env
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4           # Optional
DATABASE_PATH=data/store.db       # Optional
```

## Testing

```bash
pytest
pytest --cov=src/multi_tool_agent
```
