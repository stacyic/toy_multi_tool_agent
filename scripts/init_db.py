#!/usr/bin/env python3
"""Initialize SQLite database from CSV files."""

import csv
import sqlite3
from pathlib import Path


def initialize_database(db_path: Path, data_dir: Path) -> None:
    """
    Initialize SQLite database from CSV files.

    Args:
        db_path: Path to the SQLite database file
        data_dir: Directory containing CSV files
    """
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing database
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.executescript("""
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            phone TEXT,
            address TEXT,
            created_at TIMESTAMP
        );

        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock_level INTEGER DEFAULT 0
        );

        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date TIMESTAMP,
            total_amount REAL NOT NULL,
            status TEXT DEFAULT 'pending',
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        );

        CREATE TABLE order_items (
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            PRIMARY KEY (order_id, product_id),
            FOREIGN KEY (order_id) REFERENCES orders(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        );

        -- Create indexes for common queries
        CREATE INDEX idx_orders_customer ON orders(customer_id);
        CREATE INDEX idx_orders_date ON orders(order_date);
        CREATE INDEX idx_orders_status ON orders(status);
        CREATE INDEX idx_order_items_product ON order_items(product_id);
        CREATE INDEX idx_products_category ON products(category);
    """)

    # Load customers
    customers_csv = data_dir / "customers.csv"
    if customers_csv.exists():
        with open(customers_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute(
                    """INSERT INTO customers (id, name, email, phone, address, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        int(row["id"]),
                        row["name"],
                        row["email"],
                        row["phone"],
                        row["address"],
                        row["created_at"],
                    ),
                )
        print(f"Loaded customers from {customers_csv}")

    # Load products
    products_csv = data_dir / "products.csv"
    if products_csv.exists():
        with open(products_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute(
                    """INSERT INTO products (id, name, category, price, stock_level)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        int(row["id"]),
                        row["name"],
                        row["category"],
                        float(row["price"]),
                        int(row["stock_level"]),
                    ),
                )
        print(f"Loaded products from {products_csv}")

    # Load orders
    orders_csv = data_dir / "orders.csv"
    if orders_csv.exists():
        with open(orders_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute(
                    """INSERT INTO orders (id, customer_id, order_date, total_amount, status)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        int(row["id"]),
                        int(row["customer_id"]),
                        row["order_date"],
                        float(row["total_amount"]),
                        row["status"],
                    ),
                )
        print(f"Loaded orders from {orders_csv}")

    # Load order_items
    order_items_csv = data_dir / "order_items.csv"
    if order_items_csv.exists():
        with open(order_items_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cursor.execute(
                    """INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                       VALUES (?, ?, ?, ?)""",
                    (
                        int(row["order_id"]),
                        int(row["product_id"]),
                        int(row["quantity"]),
                        float(row["unit_price"]),
                    ),
                )
        print(f"Loaded order_items from {order_items_csv}")

    conn.commit()

    # Print summary
    cursor.execute("SELECT COUNT(*) FROM customers")
    print(f"Total customers: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM products")
    print(f"Total products: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM orders")
    print(f"Total orders: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM order_items")
    print(f"Total order_items: {cursor.fetchone()[0]}")

    conn.close()
    print(f"\nDatabase initialized at: {db_path}")


if __name__ == "__main__":
    # Default paths
    project_root = Path(__file__).parent.parent
    db_path = project_root / "data" / "store.db"
    data_dir = project_root / "data"

    initialize_database(db_path, data_dir)
