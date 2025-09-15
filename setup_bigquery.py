#!/usr/bin/env python3
"""
BigQuery setup script for M1 Multi-Source Ingestion
Creates dataset and tables from DDL schema
"""

import os
import sys
from pathlib import Path

from google.cloud import bigquery
from google.cloud.exceptions import Conflict


def get_env_vars():
    """Get required environment variables"""
    project_id = os.getenv("BQ_PROJECT", "konveyn2ai")
    dataset_id = os.getenv("BQ_DATASET", "source_ingestion")

    if not project_id:
        raise ValueError("BQ_PROJECT environment variable is required")
    if not dataset_id:
        raise ValueError("BQ_DATASET environment variable is required")

    return project_id, dataset_id


def create_dataset_if_not_exists(
    client: bigquery.Client, dataset_id: str, project_id: str
):
    """Create BigQuery dataset if it doesn't exist"""
    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)

    try:
        client.get_dataset(dataset_ref)
        print(f"✓ Dataset {project_id}.{dataset_id} already exists")
        return dataset_ref
    except Exception:
        pass

    # Create dataset
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    dataset.description = "M1 Multi-Source Ingestion for KonveyN2AI BigQuery Hackathon"
    dataset.labels = {
        "environment": "hackathon",
        "milestone": "m1",
        "purpose": "ingestion",
    }

    try:
        dataset = client.create_dataset(dataset, exists_ok=True)
        print(f"✓ Created dataset {project_id}.{dataset_id}")
        return dataset_ref
    except Conflict:
        print(f"✓ Dataset {project_id}.{dataset_id} already exists")
        return dataset_ref


def load_ddl_from_file():
    """Load DDL SQL from contracts file"""
    ddl_path = (
        Path(__file__).parent
        / "specs"
        / "002-m1-parse-and"
        / "contracts"
        / "bigquery-ddl.sql"
    )

    if not ddl_path.exists():
        raise FileNotFoundError(f"DDL file not found: {ddl_path}")

    with open(ddl_path) as f:
        ddl_content = f.read()

    return ddl_content


def substitute_variables(ddl_content: str, project_id: str, dataset_id: str):
    """Substitute environment variables in DDL"""
    ddl_content = ddl_content.replace("${BQ_PROJECT}", project_id)
    ddl_content = ddl_content.replace("${BQ_DATASET}", dataset_id)
    return ddl_content


def execute_ddl_statements(client: bigquery.Client, ddl_content: str):
    """Execute individual DDL statements"""
    # Split DDL into individual statements
    statements = []
    current_statement = []

    for line in ddl_content.split("\n"):
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("--"):
            continue

        current_statement.append(line)

        # End of statement
        if line.endswith(";"):
            statement = " ".join(current_statement).rstrip(";")
            if statement.strip():
                statements.append(statement)
            current_statement = []

    # Execute each statement
    for i, statement in enumerate(statements):
        try:
            if statement.strip().startswith("CREATE TABLE"):
                # Extract table name for better error reporting
                table_name = (
                    statement.split("`")[1].split("`")[0]
                    if "`" in statement
                    else "unknown"
                )
                print(f"Creating table {table_name}...")

            query_job = client.query(statement)
            query_job.result()  # Wait for completion

            if "CREATE TABLE" in statement:
                print("✓ Table created successfully")
            elif "ALTER SCHEMA" in statement:
                print("✓ Schema metadata updated")

        except Conflict as e:
            if "already exists" in str(e).lower():
                table_name = (
                    statement.split("`")[1].split("`")[0]
                    if "`" in statement
                    else "table"
                )
                print(f"✓ Table {table_name} already exists")
            else:
                print(f"✗ Error creating table: {e}")
                continue
        except Exception as e:
            print(f"✗ Error executing statement {i+1}: {e}")
            print(f"Statement: {statement[:100]}...")
            continue


def verify_tables(client: bigquery.Client, project_id: str, dataset_id: str):
    """Verify that all required tables exist"""
    expected_tables = ["source_metadata", "source_metadata_errors", "ingestion_log"]

    dataset_ref = bigquery.DatasetReference(project_id, dataset_id)

    existing_tables = []
    try:
        tables = client.list_tables(dataset_ref)
        existing_tables = [table.table_id for table in tables]
    except Exception as e:
        print(f"✗ Error listing tables: {e}")
        return False

    print("\nTable verification:")
    all_exist = True
    for table_name in expected_tables:
        if table_name in existing_tables:
            print(f"✓ {table_name}")
        else:
            print(f"✗ {table_name} - MISSING")
            all_exist = False

    if existing_tables:
        print(f"\nAll tables in dataset: {', '.join(existing_tables)}")

    return all_exist


def main():
    """Main setup function"""
    try:
        # Get environment variables
        project_id, dataset_id = get_env_vars()
        print(f"Setting up BigQuery for project: {project_id}, dataset: {dataset_id}")

        # Initialize BigQuery client
        client = bigquery.Client(project=project_id)
        print("✓ Connected to BigQuery")

        # Create dataset
        create_dataset_if_not_exists(client, dataset_id, project_id)

        # Load and process DDL
        print("\nLoading DDL schema...")
        ddl_content = load_ddl_from_file()
        ddl_content = substitute_variables(ddl_content, project_id, dataset_id)

        # Execute DDL statements
        print("\nExecuting DDL statements...")
        execute_ddl_statements(client, ddl_content)

        # Verify tables
        if verify_tables(client, project_id, dataset_id):
            print("\n✅ BigQuery setup completed successfully!")
            print(f"Dataset: {project_id}.{dataset_id}")
            print(f"Access via: bq ls {dataset_id}")
        else:
            print("\n⚠️  Setup completed with some missing tables")
            return 1

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
