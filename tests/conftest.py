"""Shared pytest fixtures for all tests."""

import os
import sqlite3
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def sample_schema() -> Dict[str, List[Dict]]:
    """Sample database schema for testing."""
    return {
        "customers": [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "TEXT"},
            {"name": "email", "type": "TEXT"},
            {"name": "phone", "type": "TEXT"},
            {"name": "address", "type": "TEXT"},
            {"name": "created_at", "type": "TEXT"},
        ],
        "products": [
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "TEXT"},
            {"name": "category", "type": "TEXT"},
            {"name": "price", "type": "REAL"},
            {"name": "stock_level", "type": "INTEGER"},
        ],
        "orders": [
            {"name": "id", "type": "INTEGER"},
            {"name": "customer_id", "type": "INTEGER"},
            {"name": "order_date", "type": "TEXT"},
            {"name": "total_amount", "type": "REAL"},
            {"name": "status", "type": "TEXT"},
        ],
        "order_items": [
            {"name": "order_id", "type": "INTEGER"},
            {"name": "product_id", "type": "INTEGER"},
            {"name": "quantity", "type": "INTEGER"},
            {"name": "unit_price", "type": "REAL"},
        ],
    }


@pytest.fixture
def temp_db(sample_schema) -> Path:
    """Create a temporary SQLite database with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            address TEXT,
            created_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock_level INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            order_date TEXT,
            total_amount REAL,
            status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE order_items (
            order_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            unit_price REAL,
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    # Insert sample data
    cursor.executemany(
        "INSERT INTO customers (id, name, email, phone, address, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "John Doe", "john@example.com", "555-123-4567", "123 Main St", "2024-01-01"),
            (2, "Jane Smith", "jane@example.com", "555-987-6543", "456 Oak Ave", "2024-01-15"),
            (3, "Bob Wilson", "bob@example.com", "555-456-7890", "789 Pine Rd", "2024-02-01"),
        ],
    )

    cursor.executemany(
        "INSERT INTO products (id, name, category, price, stock_level) VALUES (?, ?, ?, ?, ?)",
        [
            (1, "Laptop", "Electronics", 999.99, 50),
            (2, "Mouse", "Electronics", 29.99, 200),
            (3, "Desk Chair", "Home", 199.99, 30),
            (4, "Notebook", "Accessories", 9.99, 500),
        ],
    )

    cursor.executemany(
        "INSERT INTO orders (id, customer_id, order_date, total_amount, status) VALUES (?, ?, ?, ?, ?)",
        [
            (1, 1, "2024-06-01", 1029.98, "Delivered"),
            (2, 2, "2024-06-15", 199.99, "Shipped"),
            (3, 1, "2024-07-01", 39.98, "Placed"),
        ],
    )

    cursor.executemany(
        "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?, ?, ?, ?)",
        [
            (1, 1, 1, 999.99),
            (1, 2, 1, 29.99),
            (2, 3, 1, 199.99),
            (3, 2, 1, 29.99),
            (3, 4, 1, 9.99),
        ],
    )

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    if db_path.exists():
        os.unlink(db_path)


@pytest.fixture
def temp_policies() -> Path:
    """Create a temporary policies file."""
    content = """# Company Policies

## Return Policy

Customers may return products within 30 days of purchase.

### Category-Specific Rules
- Electronics: 15-day return window with 15% restocking fee
- Apparel: 30-day return window, items must be unworn

## Customer Tiers

### VIP Customer Definition
A VIP customer is defined as anyone who has spent over $1,000 in the last 12 months.

VIP benefits include:
- 10% discount on all purchases
- Free express shipping
- Early access to sales

## Shipping Policy

- Standard Shipping: 5-7 business days
- Express Shipping: 2-3 business days
- Free Shipping: Available on orders over $50
"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    ) as f:
        f.write(content)
        policies_path = Path(f.name)

    yield policies_path

    # Cleanup
    if policies_path.exists():
        os.unlink(policies_path)


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI API response."""
    mock_response = MagicMock()
    mock_response.content = '{"sql": "SELECT * FROM customers LIMIT 10", "explanation": "Query all customers", "columns": ["id", "name", "email"]}'
    return mock_response


@pytest.fixture
def mock_llm(mock_openai_response):
    """Mock LangChain ChatOpenAI."""
    mock = AsyncMock()
    mock.ainvoke.return_value = mock_openai_response
    return mock


@pytest.fixture
def mock_embeddings():
    """Mock OpenAI embeddings."""
    mock = MagicMock()
    # Return 1536-dimensional vectors
    mock.embed_documents.return_value = [[0.1] * 1536 for _ in range(3)]
    mock.embed_query.return_value = [0.1] * 1536
    return mock
