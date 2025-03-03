import os
import boto3
import pandas as pd
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

def test_aws_connection():
    """Test basic AWS connectivity using credentials from environment variables"""
    print("=== AWS Connection Test ===")
    print("Checking environment variables...")
    
    # Check for required environment variables
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_session_token = os.getenv("AWS_SESSION_TOKEN")
    aws_region = os.getenv("AWS_REGION", "us-east-1")
    
    if not aws_access_key or not aws_secret_key:
        print("ERROR: Missing required AWS credentials in environment variables")
        print("Required: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        print("Optional: AWS_SESSION_TOKEN, AWS_REGION")
        return False
    
    print(f"AWS_ACCESS_KEY_ID: {aws_access_key[:4]}...{aws_access_key[-4:]}")
    print(f"AWS_SECRET_ACCESS_KEY: {aws_secret_key[:4]}...{aws_secret_key[-4:]}")
    print(f"AWS_SESSION_TOKEN: {'Present' if aws_session_token else 'Not provided'}")
    print(f"AWS_REGION: {aws_region}")
    
    # Initialize session and clients
    try:
        print("\nInitializing AWS session...")
        session = boto3.Session(
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
            region_name=aws_region
        )
        
        # Test S3 connection
        print("Testing S3 connection...")
        s3_client = session.client('s3')
        buckets = s3_client.list_buckets()
        print(f"SUCCESS: Connected to S3. Found {len(buckets['Buckets'])} buckets.")
        print("Bucket names:")
        for bucket in buckets['Buckets']:
            print(f"  - {bucket['Name']}")
        
        # Test Athena connection
        print("\nTesting Athena connection...")
        athena_client = session.client('athena')
        data_catalogs = athena_client.list_data_catalogs()
        print(f"SUCCESS: Connected to Athena. Found {len(data_catalogs['DataCatalogsSummary'])} data catalogs.")
        print("Data catalogs:")
        for catalog in data_catalogs['DataCatalogsSummary']:
            print(f"  - {catalog['CatalogName']} ({catalog['Type']})")
        
        return True
        
    except Exception as e:
        print(f"ERROR: AWS connection failed: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Check if your credentials are correct and not expired")
        print("2. Verify your AWS region is correct")
        print("3. Ensure your IAM user/role has appropriate permissions")
        return False

def list_athena_databases():
    """List all Athena databases accessible with current credentials"""
    try:
        print("\n=== Athena Databases ===")
        athena_client = boto3.client('athena',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        
        # Get list of databases
        response = athena_client.list_databases(
            CatalogName='AwsDataCatalog'
        )
        
        print(f"Found {len(response['DatabaseList'])} databases:")
        for db in response['DatabaseList']:
            print(f"  - {db['Name']}")
            
        return response['DatabaseList']
        
    except Exception as e:
        print(f"ERROR: Could not list Athena databases: {str(e)}")
        return []

def list_athena_tables(database_name):
    """List all tables in the specified Athena database"""
    try:
        print(f"\n=== Tables in Database: {database_name} ===")
        athena_client = boto3.client('athena',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        
        # Get list of tables
        response = athena_client.list_table_metadata(
            CatalogName='AwsDataCatalog',
            DatabaseName=database_name
        )
        
        print(f"Found {len(response['TableMetadataList'])} tables:")
        for table in response['TableMetadataList']:
            print(f"  - {table['Name']}")
            
        return response['TableMetadataList']
        
    except Exception as e:
        print(f"ERROR: Could not list tables in {database_name}: {str(e)}")
        return []

def run_athena_query(query, database="demo"):
    """Run a query against Athena and return the results as a DataFrame"""
    print(f"\n=== Running Athena Query ===")
    print(f"Database: {database}")
    print(f"Query: {query}")
    
    try:
        athena_client = boto3.client('athena',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        
        # Set up S3 output location
        s3_output_location = os.getenv("ATHENA_OUTPUT_LOCATION", "s3://ab-destination-iceberg/")
        print(f"Query results will be stored in: {s3_output_location}")
        
        # Start query execution
        query_start_time = time.time()
        print("Starting query execution...")
        query_execution = athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': database},
            ResultConfiguration={
               'OutputLocation': s3_output_location
            }
        )
        
        query_id = query_execution['QueryExecutionId']
        print(f"Query ID: {query_id}")
        
        # Wait for completion
        print("Waiting for query to complete...")
        max_retries = 30
        attempts = 0
        
        while attempts < max_retries:
            status = athena_client.get_query_execution(QueryExecutionId=query_id)
            state = status['QueryExecution']['Status']['State']
            
            if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
                
            print(f"  Status: {state}... waiting")
            time.sleep(1)
            attempts += 1
        
        query_end_time = time.time()
        execution_time = query_end_time - query_start_time
        
        # Check final state
        if state != 'SUCCEEDED':
            error_message = status['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
            print(f"ERROR: Query failed with state: {state}")
            print(f"Reason: {error_message}")
            return None
        
        print(f"Query completed successfully in {execution_time:.2f} seconds")
        
        # Get results
        print("Fetching query results...")
        results = athena_client.get_query_results(QueryExecutionId=query_id)
        
        # Parse into DataFrame
        columns = [col['Label'] for col in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
        rows = []
        
        # Skip header row
        for row in results['ResultSet']['Rows'][1:]:
            values = [col.get('VarCharValue', '') for col in row['Data']]
            rows.append(dict(zip(columns, values)))
        
        df = pd.DataFrame(rows)
        
        print(f"Query returned {len(df)} rows and {len(columns)} columns")
        print("Column names:")
        for col in columns:
            print(f"  - {col}")
        
        return df
        
    except Exception as e:
        print(f"ERROR: Failed to execute query: {str(e)}")
        return None

def main():
    # Test AWS connection
    if not test_aws_connection():
        return
    
    # Get available databases
    databases = list_athena_databases()
    if not databases:
        return
    
    # Ask user which database to use
    db_name = input("\nEnter database name to explore (or press Enter for 'demo'): ").strip() or "demo"
    
    # List tables in the selected database
    tables = list_athena_tables(db_name)
    if not tables:
        return
    
    # Ask user which table to query
    table_name = input("\nEnter table name to query: ").strip()
    if not table_name:
        print("No table name provided. Exiting.")
        return
    
    # Run a simple query against the selected table
    query = f"SELECT * FROM {db_name}.{table_name} LIMIT 10"
    df = run_athena_query(query, db_name)
    
    if df is not None and not df.empty:
        print("\n=== First 5 rows of data ===")
        print(df.head(5))
        
        # Option to save to CSV
        save_option = input("\nWould you like to save the results to a CSV file? (y/n): ").strip().lower()
        if save_option == 'y':
            filename = f"{db_name}_{table_name}_sample.csv"
            df.to_csv(filename, index=False)
            print(f"Data saved to {filename}")

if __name__ == "__main__":
    main()