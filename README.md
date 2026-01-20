# Multi-Tool Agent

A resilient multi-tool LangChain agent with RAG, Text-to-SQL, and PII protection capabilities.

## Features

- **RAG (Retrieval-Augmented Generation)**: Understand business logic from policy documents using FAISS vector store
- **Text-to-SQL with Self-Correction**: Generate and execute SQL queries with up to 3 automatic retries
- **PII Guardrails**: Automatically mask sensitive information (phone, email, address) in responses
- **Batch Processing**: Process multiple queries efficiently with shared context optimization
- **Performance Logging**: Component-level timing and debugging for production monitoring

## Requirements

- Python 3.9 - 3.13 (Python 3.14+ has compatibility issues with langchain)
- OpenAI API key

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd multi_tool_agent

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# For development (includes pytest, black, ruff, mypy)
pip install -e ".[dev]"

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## API Key Setup

This agent requires an OpenAI API key for LLM-based query routing, SQL generation, and response synthesis.

### Option 1: Environment File (Recommended)

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=sk-your-api-key-here
```

### Option 2: Environment Variable

Export the key directly in your shell:

```bash
# Linux/macOS
export OPENAI_API_KEY=sk-your-api-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-your-api-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-your-api-key-here
```

### Option 3: Pass Programmatically

When using as a library, pass the API key directly:

```python
from multi_tool_agent.core.agent import MultiToolAgent

agent = MultiToolAgent(
    db_path="data/store.db",
    policies_path="data/policies.md",
    api_key="sk-your-api-key-here",  # Pass directly
)
```

### Verify Connection

Test that your API key is configured correctly:

```bash
.venv/bin/python3 -m multi_tool_agent.main test-connection
```

## Quick Start

```bash
# Initialize the database
.venv/bin/python3 -m multi_tool_agent.main init-db --data-dir ./data

# Single query
.venv/bin/python3 -m multi_tool_agent.main query "What is the return policy?"

# Interactive mode
.venv/bin/python3 -m multi_tool_agent.main query

# With performance logging
.venv/bin/python3 -m multi_tool_agent.main query --perf "How many VIP customers do we have?"
```

## Batch Processing

Process multiple queries efficiently with optimized API calls:

```bash
# From command line arguments
.venv/bin/python3 -m multi_tool_agent.main batch "What is the return policy?" "How many VIP customers?"

# From JSON file
.venv/bin/python3 -m multi_tool_agent.main batch --file queries.json --output results.json

# With performance logging and verbose output
.venv/bin/python3 -m multi_tool_agent.main batch --file queries.json -v --perf

# From stdin
echo '["Query 1", "Query 2"]' | .venv/bin/python3 -m multi_tool_agent.main batch
```

**Batch Optimizations:**
- Shared policy context across queries (reduces RAG API calls)
- Parallel execution of independent queries
- Aggregate metrics for batch analysis

## Performance Logging

Enable detailed component-level timing for production debugging:

```bash
# Enable with --perf flag
.venv/bin/python3 -m multi_tool_agent.main query --perf "Show me VIP customers"
```

**Output format:**
```
[PERF] request_id=abc123 | total=1234ms | SUCCESS
  ├─ routing: 45ms
  ├─ sql_execution: 890ms (attempts: 2)
  ├─ pii_masking: 2ms
  └─ synthesis: 285ms
[SQL] SELECT name, email FROM customers WHERE tier='VIP' LIMIT 10
```

## Usage as Library

```python
import asyncio
from multi_tool_agent.core.agent import MultiToolAgent
from multi_tool_agent.config import get_settings

async def main():
    settings = get_settings()
    agent = MultiToolAgent(
        db_path=settings.database.path,
        policies_path=settings.policies_path,
        model="gpt-4",
        api_key=settings.openai.api_key,
    )

    # Single query
    result = await agent.execute("What is the return policy?")
    print(result)

    # Batch processing
    queries = [
        "What is the return policy?",
        "How many VIP customers do we have?",
        "List products under $100",
    ]
    batch_result = await agent.execute_batch(queries)

    for r in batch_result:
        print(f"Query: {r.query}")
        print(f"Response: {r.response}")
        print(f"Time: {r.metrics.total_time_ms}ms")
        print()

    print(f"Batch metrics: {batch_result.batch_metrics.to_dict()}")

asyncio.run(main())
```

## Architecture

```
multi_tool_agent/
├── core/
│   ├── agent.py           # Main agent orchestration
│   ├── query_router.py    # LLM-based query routing
│   ├── batch_processor.py # Batch query optimization
│   └── exceptions.py      # Custom exceptions
├── tools/
│   ├── policy_accessor.py # RAG tool for policies
│   └── sql_accessor/
│       ├── sql_tool.py       # SQL orchestration with retry
│       ├── query_generator.py # LLM-based SQL generation
│       ├── query_checker.py   # AST-based SQL validation
│       ├── query_executor.py  # SQL execution
│       └── pii_masker.py      # PII detection and masking
├── stores/
│   └── faiss_store.py     # FAISS vector store
├── logging/
│   ├── trace_logger.py       # Request tracing and cost tracking
│   └── performance_logger.py # Component-level timing
└── main.py                # CLI entry point
```

## Configuration

Environment variables (via `.env` file):

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
OPENAI_CHAT_MODEL=gpt-4
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
DATABASE_PATH=data/store.db
LOG_DESTINATION=both  # console, file, or both
LOG_LEVEL=INFO
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/multi_tool_agent

# Run specific test file
pytest tests/unit/test_batch_processor.py -v
```