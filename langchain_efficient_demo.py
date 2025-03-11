import os
import boto3
import pandas as pd
from dotenv import load_dotenv
import time
import json
import hashlib
import pickle
import os.path
import warnings
import requests
from datetime import datetime, timedelta

# LangChain imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)

class AwsLangChainBot:
    def __init__(self, cache_dir="./cache"):
        print("Initializing Airbyte Data Chat...")
        
        # Configuration
        self.cache_dir = cache_dir
        self.vector_store_path = os.path.join(cache_dir, "faiss_index")
        self.query_cache_path = os.path.join(cache_dir, "query_cache.pickle")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Force reload environment variables to ensure we have the latest credentials
        load_dotenv(override=True)
        
        # AWS configuration from environment
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # OAuth refresh configuration
        self.aws_refresh_token = os.getenv("AWS_REFRESH_TOKEN")
        self.aws_client_id = os.getenv("AWS_CLIENT_ID")
        self.aws_client_secret = os.getenv("AWS_CLIENT_SECRET")
        self.aws_token_url = os.getenv("AWS_TOKEN_URL", "https://api.aws.amazon.com/oauth2/token")
        
        # Initialize AWS clients flag - we'll initialize them later
        self.aws_clients_initialized = False
        self.athena_client = None
        self.s3_client = None
        
        # Initialize LangChain components with updated imports
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.vector_store = None
        self.qa_chain = None
        
        # Load query cache
        self.query_cache = self.load_query_cache()
    
    def initialize_aws_clients(self):
        """Initialize AWS clients with current credentials and handle token refresh"""
        try:
            session = boto3.Session(
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                aws_session_token=self.aws_session_token,
                region_name=self.aws_region
            )
            
            self.athena_client = session.client('athena')
            self.s3_client = session.client('s3')
            
            # Test the connection with more robust error handling
            try:
                self.athena_client.list_work_groups()
                print("✅ AWS credentials are valid")
                self.aws_clients_initialized = True
                return True
            except Exception as e:
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ["expired", "token", "credentials", "access denied", "not authorized"]):
                    print("⚠️ AWS token expired during validation.")
                    # Attempt to refresh using OAuth
                    if self.refresh_aws_credentials_oauth():
                        print("✅ Credentials refreshed automatically via OAuth")
                        # Already re-initialized clients in refresh method
                        return True
                    else:
                        self.aws_clients_initialized = False
                        raise Exception("AWS credentials expired and refresh failed. Please refresh your token.")
                else:
                    raise
            
        except Exception as e:
            print(f"❌ Error initializing AWS clients: {e}")
            self.aws_clients_initialized = False
            raise Exception(f"Failed to initialize AWS clients. Please check your credentials: {str(e)}")
    
    def refresh_aws_credentials(self, access_key=None, secret_key=None, session_token=None):
        """Refresh AWS credentials and reinitialize clients"""
        try:
            # Update credentials if provided
            if access_key:
                self.aws_access_key_id = access_key
                os.environ["AWS_ACCESS_KEY_ID"] = access_key
            
            if secret_key:
                self.aws_secret_access_key = secret_key
                os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
            
            if session_token is not None:  # Allow empty string to clear token
                self.aws_session_token = session_token
                os.environ["AWS_SESSION_TOKEN"] = session_token
            
            # Reinitialize AWS clients with new credentials
            self.initialize_aws_clients()
            print("✅ AWS credentials refreshed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error refreshing AWS credentials: {e}")
            return False
    
    def refresh_aws_credentials_oauth(self, refresh_token=None):
        """Refresh AWS credentials using OAuth refresh token"""
        # Use provided refresh token or the one stored in environment
        refresh_token = refresh_token or os.getenv("AWS_REFRESH_TOKEN")
        
        if not refresh_token:
            print("❌ No refresh token available. Please provide a refresh token.")
            return False
        
        try:
            print("🔄 Refreshing AWS credentials via OAuth...")
            
            # Get the OAuth endpoint from environment or use default
            oauth_endpoint = os.getenv("AWS_OAUTH_ENDPOINT", "https://api.example.com")
            client_id = os.getenv("AWS_CLIENT_ID")
            client_secret = os.getenv("AWS_CLIENT_SECRET")
            
            # Make the OAuth refresh token request
            response = requests.post(
                f"{oauth_endpoint}/applications/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "refresh_token": refresh_token
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code != 200:
                print(f"❌ OAuth refresh failed: {response.status_code} - {response.text}")
                return False
            
            # Parse the response
            token_data = response.json()
            
            # Extract AWS credentials from the token response
            self.aws_access_key_id = token_data.get("access_key_id")
            self.aws_secret_access_key = token_data.get("secret_access_key")
            self.aws_session_token = token_data.get("session_token")
            
            # Update environment variables
            os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
            os.environ["AWS_SESSION_TOKEN"] = self.aws_session_token
            
            # Calculate expiry time (default 1 hour if not specified)
            expires_in = token_data.get("expires_in", 3600)
            self.credentials_expiry = datetime.now() + timedelta(seconds=expires_in)
            
            # Save the updated credentials
            self.save_credentials_cache()
            
            # Reinitialize AWS clients with new credentials
            self.initialize_aws_clients()
            
            print(f"✅ AWS credentials refreshed via OAuth, valid until {self.credentials_expiry}")
            return True
            
        except Exception as e:
            print(f"❌ Error refreshing AWS credentials via OAuth: {str(e)}")
            return False
    
    def update_dotenv_file(self, key, value):
        """Update a specific key in the .env file"""
        env_path = '.env'
        
        # Read current .env file
        if os.path.exists(env_path):
            with open(env_path, 'r') as file:
                lines = file.readlines()
        else:
            lines = []
        
        # Check if the key exists and update it
        key_found = False
        for i, line in enumerate(lines):
            if line.startswith(f"{key}="):
                lines[i] = f"{key}=\"{value}\"\n"
                key_found = True
                break
        
        # Add key if not found
        if not key_found:
            lines.append(f"{key}=\"{value}\"\n")
        
        # Write back to .env file
        with open(env_path, 'w') as file:
            file.writelines(lines)
    
    def load_query_cache(self):
        """Load the query cache from disk if it exists"""
        if os.path.exists(self.query_cache_path):
            try:
                with open(self.query_cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Error loading query cache: {e}")
        return {}
    
    def save_query_cache(self):
        """Save the query cache to disk"""
        try:
            with open(self.query_cache_path, 'wb') as f:
                pickle.dump(self.query_cache, f)
        except Exception as e:
            print(f"Error saving query cache: {e}")
    
    def get_cache_key(self, query, database):
        """Generate a cache key for the query"""
        key = f"{query}_{database}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def run_athena_query(self, query, database="demo", catalog_name="AwsDataCatalog"):
        """Run Athena query and return DataFrame with caching and automatic OAuth refresh"""
        cache_key = self.get_cache_key(query, database)
        
        # Check if this query is already in cache
        if cache_key in self.query_cache:
            print(f"📋 Using cached results for query: {query[:50]}...")
            return self.query_cache[cache_key]
        
        print(f"🔍 Running Athena query: {query[:50]}...")
            
        # Create S3 output location if not specified in environment
        s3_output_location = os.getenv("ATHENA_OUTPUT_LOCATION", "s3://ab-destination-iceberg/")
        
        try:
            # Make sure AWS clients are initialized
            if not self.aws_clients_initialized:
                self.initialize_aws_clients()
            
            # Verify the database exists before running the query
            try:
                databases = self.athena_client.list_databases(
                    CatalogName=catalog_name
                )
                db_names = [db['Name'] for db in databases['DatabaseList']]
                if database not in db_names:
                    print(f"⚠️ Database '{database}' not found. Available databases: {', '.join(db_names)}")
                    raise Exception(f"Database '{database}' does not exist")
            except Exception as e:
                if "expired" in str(e).lower() or "token" in str(e).lower() or "credentials" in str(e).lower():
                    if self.refresh_aws_credentials_oauth():
                        print("✅ Credentials refreshed automatically via OAuth")
                        # Retry the query after refreshing
                        return self.run_athena_query(query, database, catalog_name)
                    else:
                        raise Exception("AWS credentials expired and refresh failed.")
                raise
            
            # Start the query execution
            query_execution = self.athena_client.start_query_execution(
                QueryString=query,
                QueryExecutionContext={
                    'Database': database,
                    'Catalog': catalog_name
                },
                ResultConfiguration={
                   'OutputLocation': s3_output_location
                }
            )
            
            query_id = query_execution['QueryExecutionId']
            
            # Wait for completion
            max_retries = 30
            attempts = 0
            while attempts < max_retries:
                try:
                    status = self.athena_client.get_query_execution(QueryExecutionId=query_id)
                    state = status['QueryExecution']['Status']['State']
                    if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                        break
                    print("⏳ Waiting for query to complete...")
                    time.sleep(1)
                    attempts += 1
                except Exception as e:
                    error_str = str(e).lower()
                    # Check for token expiration with more possible error messages
                    if any(keyword in error_str for keyword in ["expired", "token", "credentials", "access denied", "not authorized"]):
                        print("⚠️ AWS token expired during query execution. Attempting to refresh...")
                        if self.refresh_aws_credentials_oauth():
                            print("✅ Credentials refreshed, continuing with query...")
                            continue
                        else:
                            raise Exception("AWS credentials expired and refresh failed.")
                    else:
                        raise
            
            if state != 'SUCCEEDED':
                error_message = status['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                print(f"❌ Query failed with state: {state}, reason: {error_message}")
                raise Exception(f"Athena query failed: {error_message}")
            
            # Get results
            try:
                results = self.athena_client.get_query_results(QueryExecutionId=query_id)
            except Exception as e:
                # Check if it's a token expiration error
                if "expired" in str(e).lower() or "token" in str(e).lower():
                    print("⚠️ AWS token expired while getting results. Attempting to refresh...")
                    if self.refresh_aws_credentials_oauth():
                        print("✅ Credentials refreshed, retrying...")
                        return self.run_athena_query(query, database, catalog_name)
                    else:
                        raise Exception("AWS credentials expired and refresh failed.")
                else:
                    raise
            
            # Parse into DataFrame
            columns = [col['Label'] for col in results['ResultSet']['ResultSetMetadata']['ColumnInfo']]
            rows = []
            for row in results['ResultSet']['Rows'][1:]:  # Skip header
                values = [col.get('VarCharValue', '') for col in row['Data']]
                rows.append(dict(zip(columns, values)))
            
            df = pd.DataFrame(rows)
            print(f"✅ Query returned {len(df)} rows")
            
            # Cache the results
            self.query_cache[cache_key] = df
            self.save_query_cache()
            
            return df
        except Exception as e:
            # Check if it's a token expiration error and attempt to refresh
            if "expired" in str(e).lower() or "token" in str(e).lower() or "credentials" in str(e).lower():
                print("⚠️ AWS token expired. Attempting to refresh via OAuth...")
                if self.refresh_aws_credentials_oauth():
                    print("✅ Credentials refreshed, retrying query...")
                    # Retry the query with refreshed credentials
                    return self.run_athena_query(query, database, catalog_name)
                else:
                    print("❌ OAuth refresh failed.")
            
            print(f"❌ Error executing Athena query: {e}")
            raise Exception(f"Failed to execute Athena query: {str(e)}")
    
    def fetch_and_process_data(self):
        """Fetch data from AWS Glue/Athena and convert to Documents for embedding"""
        print("📥 Fetching data from AWS data lake...")
        
        try:
            # Try to fetch data from AWS
            try:
                # Test AWS connection
                self.athena_client.list_work_groups()
                aws_available = True
            except Exception as e:
                print(f"⚠️ AWS connection error: {str(e)}")
                # Try to refresh credentials
                if "expired" in str(e).lower() or "token" in str(e).lower():
                    if self.refresh_aws_credentials_oauth():
                        print("✅ Credentials refreshed, trying again...")
                        try:
                            self.athena_client.list_work_groups()
                            aws_available = True
                        except Exception as e2:
                            print(f"⚠️ Still can't connect to AWS after refresh: {str(e2)}")
                            aws_available = False
                    else:
                        print("⚠️ OAuth refresh failed. Using sample documents for testing.")
                        aws_available = False
            
            if aws_available:
                # Check if the required tables exist
                try:
                    tables = self.athena_client.list_table_metadata(
                        CatalogName='AwsDataCatalog',
                        DatabaseName='demo'
                    )
                    table_names = [table['Name'] for table in tables['TableMetadataList']]
                    
                    if 'files' not in table_names:
                        print("⚠️ 'files' table not found in demo database. Available tables: " + ", ".join(table_names))
                        print("⚠️ Using sample data instead.")
                        aws_available = False
                    elif 'contact' not in table_names:
                        print("⚠️ 'contact' table not found in demo database. Available tables: " + ", ".join(table_names))
                        print("⚠️ Using sample data instead.")
                        aws_available = False
                except Exception as e:
                    print(f"⚠️ Error checking tables: {str(e)}")
                    # Try to refresh credentials
                    if "expired" in str(e).lower() or "token" in str(e).lower():
                        if self.refresh_aws_credentials_oauth():
                            print("✅ Credentials refreshed, trying again...")
                            try:
                                tables = self.athena_client.list_table_metadata(
                                    CatalogName='AwsDataCatalog',
                                    DatabaseName='demo'
                                )
                                table_names = [table['Name'] for table in tables['TableMetadataList']]
                                
                                if 'files' not in table_names or 'contact' not in table_names:
                                    print("⚠️ Required tables not found. Using sample data instead.")
                                    aws_available = False
                            except Exception as e2:
                                print(f"⚠️ Still can't check tables after refresh: {str(e2)}")
                                aws_available = False
                        else:
                            print("⚠️ OAuth refresh failed. Using sample data instead.")
                            aws_available = False
                
                if aws_available:
                    # Fetch contract data from files table
                    print("📄 Fetching contract data from 'files' table...")
                    contracts_query = "SELECT * FROM demo.files"
                    contracts_df = self.run_athena_query(contracts_query, database="demo")
                    
                    # Fetch contact data from contact table
                    print("👤 Fetching contact data from 'contact' table...")
                    contacts_query = "SELECT * FROM demo.contact"
                    contacts_df = self.run_athena_query(contacts_query, database="demo")
                    
                    # Check if we have data
                    if contracts_df.empty:
                        print("⚠️ No contract data found in 'files' table. Using sample data instead.")
                        aws_available = False
                    
                    if contacts_df.empty and aws_available:
                        print("⚠️ No contact data found in 'contact' table. Using sample data instead.")
                        aws_available = False
            
            if not aws_available:
                # Create sample data for testing
                print("📝 Creating sample test data...")
                contracts_df = pd.DataFrame([
                    {
                        'filename': 'Contract_ABC_Corp.pdf',
                        'content': 'This is a contract between ABC Corp and our company for software services.',
                        'company': 'ABC Corp'
                    },
                    {
                        'filename': 'Contract_XYZ_Inc.pdf',
                        'content': 'Agreement between XYZ Inc and our company for consulting services.',
                        'company': 'XYZ Inc'
                    }
                ])
                
                contacts_df = pd.DataFrame([
                    {
                        'FirstName': 'John',
                        'LastName': 'Smith',
                        'Email': 'john.smith@abccorp.com',
                        'Phone': '555-123-4567',
                        'Company': 'ABC Corp'
                    },
                    {
                        'FirstName': 'Jane',
                        'LastName': 'Doe',
                        'Email': 'jane.doe@xyzinc.com',
                        'Phone': '555-987-6543',
                        'Company': 'XYZ Inc'
                    }
                ])
            
            # Create Document objects
            documents = []
            
            # Process contracts
            print(f"🔄 Processing {len(contracts_df)} contracts...")
            for i, row in contracts_df.iterrows():
                content = f"CONTRACT: {row.get('filename', f'Document{i}')}\n"
                for col in contracts_df.columns:
                    value = row.get(col, '')
                    # Skip binary or very large content
                    if isinstance(value, str) and len(value) < 5000:
                        content += f"{col}: {value}\n"
                
                # Add any contract-specific information
                if 'content' in row:
                    content += f"Document Content: {row['content']}\n"
                
                doc = Document(
                    page_content=content,
                    metadata={"source": "google_drive", "type": "contract", "id": str(i)}
                )
                documents.append(doc)
            
            # Process contacts - Enhance with better company name normalization
            print(f"🔄 Processing {len(contacts_df)} contacts...")
            for i, row in contacts_df.iterrows():
                # Try to find name fields with different possible column names
                first_name = None
                for col in ['FirstName', 'firstname', 'first_name', 'fname']:
                    if col in contacts_df.columns and row.get(col):
                        first_name = row.get(col)
                        break
                
                last_name = None
                for col in ['LastName', 'lastname', 'last_name', 'lname']:
                    if col in contacts_df.columns and row.get(col):
                        last_name = row.get(col)
                        break
                
                # Extract company with more variations and normalize
                company = None
                for col in ['Company', 'company', 'CompanyName', 'company_name', 'AccountName', 'account_name', 'Account']:
                    if col in contacts_df.columns and row.get(col):
                        company = str(row.get(col)).strip().lower()
                        break
                
                # Normalize company names (handle variations like "Inc", "Inc.", "Incorporated")
                if company:
                    # Remove common suffixes for matching
                    company = company.replace("inc.", "").replace("inc", "").replace("llc", "").replace("llc.", "")
                    company = company.replace("ltd.", "").replace("ltd", "").replace("limited", "")
                    company = company.replace("corp.", "").replace("corp", "").replace("corporation", "")
                    company = company.strip()
                
                name = f"{first_name or ''} {last_name or ''}".strip() or f"Contact{i}"
                
                content = f"CONTACT: {name}\n"
                if company:
                    content += f"Company: {row.get('Company', company)}\n"  # Use original company name in content
                
                for col in contacts_df.columns:
                    value = row.get(col, '')
                    # Skip binary or very large content and already processed fields
                    if isinstance(value, str) and len(value) < 5000 and col not in ['FirstName', 'LastName', 'Company']:
                        content += f"{col}: {value}\n"
                
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": "salesforce", 
                        "type": "contact", 
                        "id": str(i),
                        "company": company,  # Store normalized company for matching
                        "name": name
                    }
                )
                documents.append(doc)
            
            # Find relationships between contracts and contacts
            print("🔄 Finding relationships between contracts and contacts...")
            
            # Add relationship docs (if company names match)
            company_to_contract = {}
            company_to_contact = {}
            
            # Extract company names from contracts
            for i, row in contracts_df.iterrows():
                company = None
                for col in ['company', 'Company', 'company_name', 'CompanyName']:
                    if col in contracts_df.columns and row.get(col):
                        company = str(row.get(col)).lower()
                        break
                
                if company:
                    if company not in company_to_contract:
                        company_to_contract[company] = []
                    company_to_contract[company].append(str(i))
            
            # Extract company from contacts
            for i, row in contacts_df.iterrows():
                company = None
                for col in ['Company', 'company', 'CompanyName', 'company_name']:
                    if col in contacts_df.columns and row.get(col):
                        company = str(row.get(col)).lower()
                        break
                
                if company:
                    if company not in company_to_contact:
                        company_to_contact[company] = []
                    company_to_contact[company].append(str(i))
            
            # Create relationship documents for matching companies
            print(f"📊 Found {len(company_to_contract)} companies in contracts and {len(company_to_contact)} in contacts")
            relationship_count = 0
            
            for company in company_to_contract:
                if company in company_to_contact:
                    for contract_id in company_to_contract[company]:
                        for contact_id in company_to_contact[company]:
                            rel_content = f"RELATIONSHIP: Company '{company}' links contract {contract_id} to contact {contact_id}\n"
                            
                            # Add contract filename
                            contract_row = contracts_df.iloc[int(contract_id)]
                            if 'filename' in contract_row:
                                rel_content += f"Contract Filename: {contract_row['filename']}\n"
                            
                            # Add contact name
                            contact_row = contacts_df.iloc[int(contact_id)]
                            contact_name = ""
                            for name_col in ['FirstName', 'LastName', 'name']:
                                if name_col in contact_row and contact_row.get(name_col):
                                    if not contact_name:
                                        contact_name = contact_row[name_col]
                                    else:
                                        contact_name += f" {contact_row[name_col]}"
                            
                            if contact_name:
                                rel_content += f"Contact Name: {contact_name}\n"
                            
                            rel_doc = Document(
                                page_content=rel_content,
                                metadata={
                                    "source": "relationship",
                                    "type": "relationship",
                                    "company": company,
                                    "contract_id": contract_id,
                                    "contact_id": contact_id
                                }
                            )
                            documents.append(rel_doc)
                            relationship_count += 1
            
            print(f"✅ Created {relationship_count} relationship documents")
            
            # Split documents
            print(f"🔄 Splitting {len(documents)} documents into chunks...")
            split_docs = self.splitter.split_documents(documents)
            print(f"✅ Created {len(split_docs)} text chunks")
            
            return split_docs
            
        except Exception as e:
            print(f"❌ Error processing data: {str(e)}")
            raise
    
    def load_or_create_vector_store(self):
        """Load existing vector store or create a new one"""
        # Check if vector store exists
        if os.path.exists(os.path.join(self.vector_store_path, "index.faiss")) and \
           os.path.exists(os.path.join(self.vector_store_path, "index.pkl")):
            print("📂 Loading existing vector store...")
            try:
                # Add error handling and debugging
                print(f"Loading from path: {self.vector_store_path}")
                # Try with allow_dangerous_deserialization
                self.vector_store = FAISS.load_local(
                    self.vector_store_path, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                print("✅ Vector store loaded successfully")
                return True
            except Exception as e:
                print(f"❌ Error loading vector store: {e}")
                # Print more detailed error information
                import traceback
                traceback.print_exc()
                print("Creating a new vector store...")
        
        # Create new vector store
        try:
            documents = self.fetch_and_process_data()
            
            if not documents:
                print("❌ No documents were created. Please check your AWS data.")
                return False
            
            print(f"🔄 Creating vector store with {len(documents)} documents...")
            # Add debugging for document content
            print(f"Sample document content: {documents[0].page_content[:100]}...")
            
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
            
            # Save to disk
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
            print("✅ Vector store created and saved")
            return True
            
        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            # Print more detailed error information
            import traceback
            traceback.print_exc()
            return False
    
    def initialize(self, force_refresh=False):
        """Set up the vector store and QA chain"""
        if force_refresh:
            # Delete existing vector store
            import shutil
            if os.path.exists(self.vector_store_path):
                shutil.rmtree(self.vector_store_path)
                print("🔄 Deleted existing vector store for refresh")
        
        # Initialize AWS clients if needed
        if not self.aws_clients_initialized:
            try:
                self.initialize_aws_clients()
            except Exception as e:
                print(f"⚠️ AWS initialization error: {str(e)}")
                # Try to refresh the credentials via OAuth if token expired
                if "expired" in str(e).lower() or "token" in str(e).lower():
                    if self.refresh_aws_credentials_oauth():
                        print("✅ Credentials refreshed via OAuth. Continuing initialization.")
                    else:
                        print("⚠️ Could not refresh credentials via OAuth. Will use sample data.")
                else:
                    print("⚠️ Will proceed with sample data.")
        
        # Load or create vector store
        if not self.load_or_create_vector_store():
            print("❌ Failed to initialize vector store")
            return False
        
        # Create QA chain
        print("🔄 Setting up the conversational retrieval chain...")
        try:
            # Add more debugging and error handling
            print(f"Vector store type: {type(self.vector_store)}")
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            print(f"Retriever created: {type(retriever)}")
            
            # Create a custom memory that knows which key to use
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key="answer"  # This tells memory which output to store
            )
            
            # Create the chain with the updated memory
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0, openai_api_key=self.openai_api_key, model="gpt-4"),
                retriever=retriever,
                memory=self.memory,
                verbose=True,
                return_source_documents=True,
                output_key="answer"  # This tells the chain which key to use as the main output
            )
            
            print("✅ Initialization complete - ask questions about contracts and contacts!")
            return True
        except Exception as e:
            print(f"❌ Error creating QA chain: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def ask(self, question):
        """Ask a question about the data with improved context for relationship queries"""
        if not self.qa_chain:
            print("❌ System not initialized. Call initialize() first.")
            return "System not initialized. Please initialize first."
        
        print(f"❓ Question: {question}")
        
        try:
            # Analyze the question type
            question_lower = question.lower()
            
            # Detect relationship and analytical questions
            is_relationship_query = any(phrase in question_lower for phrase in [
                "associated with", "related to", "connected to", "linked with",
                "which company", "which contact", "who works", "employed by"
            ])
            
            is_analytical_query = any(phrase in question_lower for phrase in [
                "highest", "lowest", "most", "least", "average", "total", 
                "when did", "effective date", "start date", "end date",
                "contract fee", "value", "amount", "cost", "price"
            ])
            
            # Get more documents for relationship and analytical queries
            k_value = 20 if is_relationship_query or is_analytical_query else 10
            
            # Get documents with the enhanced retriever
            docs = self.vector_store.similarity_search(
                question,
                k=k_value
            )
            
            # Create context from documents
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Add detailed data structure guidance based on your actual data
            data_structure_guidance = """
            DATA STRUCTURE INFORMATION:
            
            1. Salesforce Contacts:
               - Primary fields: name, email, phone, title
               - Each contact has a unique ID (_airbyte_generation_id)
               - Contacts may be associated with companies but this relationship needs to be inferred
            
            2. Google Drive Contracts:
               - Primary fields: company name, effective date, contract fee/value
               - Contract details include start dates, end dates, and financial terms
               - Contracts are associated with companies by name
            
            3. Relationships:
               - Contacts and contracts are related through company names
               - A company may have multiple contacts (employees)
               - A company may have multiple contracts
               - Contract effective dates and values can be used for financial analysis
            """
            
            # Create formatting instructions based on question type
            formatting_instructions = ""
            
            if is_relationship_query:
                formatting_instructions = """
                Format your response as follows:
                
                1. Start with a direct answer to the relationship question.
                
                2. For company-contact relationships:
                   ## Contacts at [Company]
                   1. [Name]
                      - Email: [Email]
                   2. [Next Contact]
                      ...
                
                3. For company-contract relationships:
                   ## Contracts with [Company]
                   1. Contract: [Contract Name/ID]
                      - Effective Date: [Date if available]
                      - Value: [Amount if available]
                   2. [Next Contract]
                      ...
                
                Only include information that is present in the context.
                """
            elif is_analytical_query:
                formatting_instructions = """
                Format your response as follows:
                
                1. Start with a direct answer to the analytical question (e.g., "The highest contract value is $X for Company Y").
                
                2. Provide supporting details:
                   ## Analysis Details
                   - [Key finding 1]
                   - [Key finding 2]
                   ...
                
                3. If relevant, include a breakdown:
                   ## Data Breakdown
                   1. [Company]: [Relevant value/date]
                   2. [Company]: [Relevant value/date]
                   ...
                
                Be precise with dates, amounts, and numerical values. Only include information that is present in the context.
                """
            else:
                formatting_instructions = """
                Format your response in a clear, structured way with appropriate headings and bullet points where relevant.
                
                For contacts, include:
                - Name
                - Email
                
                For contracts, include:
                - Company name
                - Effective dates (if available)
                - Contract values (if available)
                
                Only include information that is present in the context.
                """
            
            # Create the full prompt
            custom_prompt = f"""
            You are an assistant that analyzes relationships between contacts and contracts in a corporate database.
            
            {data_structure_guidance}
            
            Context information is below.
            ---------------------
            {context}
            ---------------------
            
            Given the context information and not prior knowledge, answer this question: {question}
            
            {formatting_instructions}
            
            If you don't have enough information to answer the question completely, simply state what you know based on the available data without mentioning "Missing Information" explicitly. Never include a "Missing Information" section in your response.
            """
            
            # Use a direct LLM call for better control
            llm = ChatOpenAI(temperature=0, openai_api_key=self.openai_api_key, model="gpt-4")
            response = llm.invoke(custom_prompt)
            
            # Update memory
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(response.content)
            
            print(f"✅ Response generated for {'relationship' if is_relationship_query else 'analytical' if is_analytical_query else 'standard'} query")
            return response.content
            
        except Exception as e:
            error_str = str(e).lower()
            # Error handling code
            if any(keyword in error_str for keyword in ["expired", "token", "credentials", "access denied", "not authorized"]):
                print("⚠️ AWS token expired. Attempting to refresh via OAuth...")
                if self.refresh_aws_credentials_oauth():
                    print("✅ Credentials refreshed, retrying query...")
                    return self.ask(question)
                else:
                    print("❌ OAuth refresh failed.")
            else:
                print(f"❌ Error generating answer: {str(e)}")
                import traceback
                traceback.print_exc()
                return f"Error: {str(e)}"

    def explore_data(self):
        """Explore available tables and data in the connected database"""
        try:
            # List available databases
            databases = self.athena_client.list_databases()
            print("\n=== Available Databases ===")
            for db in databases['DatabaseList']:
                print(f"- {db['Name']}")
            
            # Ask which database to explore
            database = input("\nEnter database name to explore (default: demo): ") or "demo"
            
            # List tables in the selected database
            tables = self.athena_client.list_table_metadata(
                CatalogName='AwsDataCatalog',
                DatabaseName=database
            )
            
            print(f"\n=== Tables in {database} ===")
            for table in tables['TableMetadataList']:
                print(f"- {table['Name']}")
            
            # Ask which table to preview
            table = input("\nEnter table name to preview: ")
            if table:
                # Preview the table
                query = f"SELECT * FROM {database}.{table} LIMIT 10"
                result_df = self.run_athena_query(query, database=database)
                
                print(f"\n=== Preview of {database}.{table} ===")
                print(result_df.to_string())
                
                # Show column info
                print("\n=== Column Information ===")
                for col in result_df.columns:
                    print(f"- {col}: {result_df[col].dtype}")
                
                # Count rows
                count_query = f"SELECT COUNT(*) as row_count FROM {database}.{table}"
                count_df = self.run_athena_query(count_query, database=database)
                print(f"\nTotal rows: {count_df['row_count'].iloc[0]}")
            
            return True
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["expired", "token", "credentials", "access denied", "not authorized"]):
                print("⚠️ AWS credentials expired. Attempting to refresh via OAuth...")
                if self.refresh_aws_credentials_oauth():
                    print("✅ Credentials refreshed, retrying operation...")
                    return self.explore_data()
                else:
                    print("❌ Failed to refresh credentials.")
            else:
                print(f"Error exploring data: {str(e)}")
            return False

    def run_diagnostics(self, database="demo", tables=None):
        """Run comprehensive diagnostics on AWS data access"""
        if tables is None:
            tables = ["files", "contact"]
        
        print("\n======= AWS DATA ACCESS DIAGNOSTICS =======\n")
        print("This tool will check your AWS Athena and S3 connectivity and data access")
        
        # Step 1: Check AWS credentials
        print("\n--- Step 1: AWS Credential Check ---")
        try:
            catalogs = self.athena_client.list_work_groups()
            print(f"✅ AWS credentials are valid - found {len(catalogs['WorkGroups'])} work groups")
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["expired", "token", "credentials", "access denied", "not authorized"]):
                print("⚠️ AWS credentials expired. Attempting to refresh via OAuth...")
                if self.refresh_aws_credentials_oauth():
                    print("✅ Credentials refreshed, retrying diagnostics...")
                    return self.run_diagnostics(database, tables)
                else:
                    print("❌ Failed to refresh credentials.")
            else:
                print(f"❌ AWS credential check failed: {e}")
            return False
        
        # Step 2: S3 bucket access check
        print("\n--- Step 2: S3 Access Check ---")
        s3_output_location = os.getenv("ATHENA_OUTPUT_LOCATION", "s3://ab-destination-iceberg/")
        try:
            # Parse S3 URI into bucket and prefix
            s3_uri = s3_output_location
            if s3_uri.startswith('s3://'):
                s3_uri = s3_uri[5:]  # Remove s3:// prefix
            
            # Split into bucket and prefix
            parts = s3_uri.split('/', 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ''
            
            print(f"Bucket: {bucket}")
            print(f"Prefix: {prefix}")
            
            # Test bucket existence
            self.s3_client.head_bucket(Bucket=bucket)
            print(f"✅ S3 bucket '{bucket}' exists and is accessible")
            
            # Test write permissions by creating a test file
            test_key = f"{prefix}/athena_test_{int(time.time())}.txt"
            self.s3_client.put_object(
                Bucket=bucket,
                Key=test_key,
                Body="This is a test file to verify write permissions."
            )
            print(f"✅ Successfully wrote test file to {bucket}/{test_key}")
            
            # Clean up the test file
            self.s3_client.delete_object(Bucket=bucket, Key=test_key)
            print(f"✅ Successfully deleted test file")
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["expired", "token", "credentials", "access denied", "not authorized"]):
                print("⚠️ AWS credentials expired. Attempting to refresh via OAuth...")
                if self.refresh_aws_credentials_oauth():
                    print("✅ Credentials refreshed, retrying S3 check...")
                    # Rerun just this part
                    self.run_diagnostics(database, tables)
                    return True
                else:
                    print("❌ Failed to refresh credentials.")
            else:
                print(f"❌ S3 access check failed: {e}")
        
        # Step 3: Database existence check
        print(f"\n--- Step 3: Database Check ({database}) ---")
        try:
            databases = self.athena_client.list_databases()
            db_names = [db['Name'] for db in databases['DatabaseList']]
            
            if database in db_names:
                print(f"✅ Database '{database}' exists")
                
                # Step 4: Table existence and data check
                print(f"\n--- Step 4: Table Checks ---")
                all_tables_ok = True
                
                for table in tables:
                    print(f"\nChecking table '{table}':")
                    try:
                        # Check table existence
                        tables_resp = self.athena_client.list_table_metadata(
                            CatalogName='AwsDataCatalog',
                            DatabaseName=database
                        )
                        table_names = [t['Name'] for t in tables_resp['TableMetadataList']]
                        
                        if table in table_names:
                            print(f"✅ Table '{table}' exists")
                            
                            # Try to count rows to check query execution
                            count_query = f"SELECT COUNT(*) as row_count FROM {database}.{table}"
                            try:
                                result = self.run_athena_query(count_query, database)
                                if result is not None and not result.empty:
                                    row_count = result['row_count'].iloc[0]
                                    print(f"✅ Table has {row_count} rows")
                                    
                                    # Preview data if there are rows
                                    if int(row_count) > 0:
                                        preview_query = f"SELECT * FROM {database}.{table} LIMIT 3"
                                        preview = self.run_athena_query(preview_query, database)
                                        print("✅ Sample data preview:")
                                        print(preview.head(3))
                                    else:
                                        print("⚠️ Table exists but has no data")
                                        all_tables_ok = False
                                else:
                                    print("❌ Could not get row count - query failed")
                                    all_tables_ok = False
                            except Exception as e:
                                error_str = str(e).lower()
                                if any(keyword in error_str for keyword in ["expired", "token", "credentials"]):
                                    print("⚠️ AWS credentials expired. Attempting to refresh...")
                                    if self.refresh_aws_credentials_oauth():
                                        print("✅ Credentials refreshed, retrying query...")
                                        try:
                                            result = self.run_athena_query(count_query, database)
                                            row_count = result['row_count'].iloc[0]
                                            print(f"✅ Table has {row_count} rows after credential refresh")
                                        except Exception as e2:
                                            print(f"❌ Error counting rows after refresh: {e2}")
                                            all_tables_ok = False
                                    else:
                                        print("❌ Failed to refresh credentials.")
                                        all_tables_ok = False
                                else:
                                    print(f"❌ Error counting rows: {e}")
                                    all_tables_ok = False
                        else:
                            print(f"❌ Table '{table}' not found in database '{database}'")
                            print(f"   Available tables: {', '.join(table_names)}")
                            all_tables_ok = False
                    except Exception as e:
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ["expired", "token", "credentials", "access denied"]):
                            print("⚠️ AWS credentials expired during table check. Attempting to refresh...")
                            if self.refresh_aws_credentials_oauth():
                                print("✅ Credentials refreshed, continuing...")
                                continue
                            else:
                                print("❌ Failed to refresh credentials.")
                        print(f"❌ Error checking table '{table}': {e}")
                        all_tables_ok = False
                
                if all_tables_ok:
                    print("\n✅ All target tables exist and are accessible")
                else:
                    print("\n⚠️ Some tables have issues - see details above")
            else:
                print(f"❌ Database '{database}' not found")
                print(f"   Available databases: {', '.join(db_names)}")
                return False
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["expired", "token", "credentials", "access denied"]):
                print("⚠️ AWS credentials expired. Attempting to refresh...")
                if self.refresh_aws_credentials_oauth():
                    print("✅ Credentials refreshed, retrying database check...")
                    return self.run_diagnostics(database, tables)
                else:
                    print("❌ Failed to refresh credentials.")
            else:
                print(f"❌ Error checking databases: {e}")
            return False
        
        print("\n======= DIAGNOSTICS COMPLETE =======")
        return True

    def discover_tables(self):
        """Discover available tables in AWS Athena"""
        try:
            # List all databases
            print("Discovering available databases...")
            databases = self.athena_client.list_databases(CatalogName='AwsDataCatalog')
            db_names = [db['Name'] for db in databases['DatabaseList']]
            print(f"Found databases: {', '.join(db_names)}")
            
            # For each database, list tables
            for db in db_names:
                print(f"\nExploring database: {db}")
                try:
                    tables = self.athena_client.list_table_metadata(
                        CatalogName='AwsDataCatalog',
                        DatabaseName=db
                    )
                    table_names = [table['Name'] for table in tables['TableMetadataList']]
                    
                    if table_names:
                        print(f"Tables in {db}: {', '.join(table_names)}")
                        
                        # Check the first few tables to see if they have data
                        for table in table_names[:3]:  # Check first 3 tables
                            try:
                                count_query = f"SELECT COUNT(*) as row_count FROM {db}.{table}"
                                result = self.run_athena_query(count_query, database=db, catalog_name='AwsDataCatalog')
                                if result is not None and not result.empty:
                                    row_count = result['row_count'].iloc[0]
                                    print(f"- Table {db}.{table} has {row_count} rows")
                                    
                                    if int(row_count) > 0:
                                        # Get a preview of the data
                                        preview_query = f"SELECT * FROM {db}.{table} LIMIT 2"
                                        preview = self.run_athena_query(preview_query, database=db, catalog_name='AwsDataCatalog')
                                        if preview is not None and not preview.empty:
                                            print(f"- Preview of {db}.{table}:")
                                            print(preview.head(2))
                                            print(f"- Columns: {', '.join(preview.columns)}")
                            except Exception as e:
                                error_str = str(e).lower()
                                if any(keyword in error_str for keyword in ["expired", "token", "credentials"]):
                                    print(f"⚠️ AWS credentials expired while accessing {db}.{table}. Attempting to refresh...")
                                    if self.refresh_aws_credentials_oauth():
                                        print("✅ Credentials refreshed, continuing...")
                                        continue
                                    else:
                                        print("❌ Failed to refresh credentials.")
                                        break
                                print(f"- Error accessing {db}.{table}: {str(e)}")
                    else:
                        print(f"No tables found in database {db}")
                except Exception as e:
                    error_str = str(e).lower()
                    if any(keyword in error_str for keyword in ["expired", "token", "credentials"]):
                        print(f"⚠️ AWS credentials expired while listing tables in {db}. Attempting to refresh...")
                        if self.refresh_aws_credentials_oauth():
                            print("✅ Credentials refreshed, retrying...")
                            try:
                                tables = self.athena_client.list_table_metadata(
                                    CatalogName='AwsDataCatalog',
                                    DatabaseName=db
                                )
                                table_names = [table['Name'] for table in tables['TableMetadataList']]
                                print(f"Tables in {db} after refresh: {', '.join(table_names)}")
                            except Exception as e2:
                                print(f"Error listing tables in {db} after refresh: {str(e2)}")
                        else:
                            print("❌ Failed to refresh credentials.")
                            continue
                    else:
                        print(f"Error listing tables in {db}: {str(e)}")
            
            return True
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["expired", "token", "credentials"]):
                print("⚠️ AWS credentials expired. Attempting to refresh...")
                if self.refresh_aws_credentials_oauth():
                    print("✅ Credentials refreshed, retrying table discovery...")
                    return self.discover_tables()
                else:
                    print("❌ Failed to refresh credentials.")
            else:
                print(f"Error discovering tables: {str(e)}")
            return False

# Streamlit interface
def create_streamlit_app():
    import streamlit as st
    
    st.set_page_config(
        page_title="Airbyte Data Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Airbyte Data Chatbot")
    st.subheader("Ask questions about your contracts and contacts")
    
    # Initialize bot in session state
    if 'bot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            st.session_state.bot = AwsLangChainBot()
            st.session_state.bot.initialize()
    
    # Add data inspection tab in sidebar
    st.sidebar.title("Data Tools")
    
    # Add AWS authentication section
    with st.sidebar.expander("AWS Authentication"):
        st.subheader("Update AWS Credentials")
        
        # Show current status
        if hasattr(st.session_state.bot, 'aws_access_key_id') and st.session_state.bot.aws_access_key_id:
            st.success(f"AWS Access Key: {st.session_state.bot.aws_access_key_id[:5]}...")
        else:
            st.warning("No AWS access key detected")
            
        # Option for OAuth refresh
        st.subheader("OAuth Refresh")
        refresh_token = st.text_input("AWS Refresh Token (leave empty to use stored token)",
                                    type="password", key="refresh_token")
        
        if st.button("Refresh AWS Credentials via OAuth"):
            with st.spinner("Refreshing AWS credentials..."):
                token_to_use = refresh_token if refresh_token else None
                if st.session_state.bot.refresh_aws_credentials_oauth(token_to_use):
                    st.success("✅ AWS credentials refreshed successfully via OAuth!")
                else:
                    st.error("❌ Failed to refresh AWS credentials via OAuth")
        
        # Manual credential update option
        st.subheader("Manual Credential Update")
        access_key = st.text_input("AWS Access Key ID", type="password", key="access_key")
        secret_key = st.text_input("AWS Secret Access Key", type="password", key="secret_key")
        session_token = st.text_input("AWS Session Token", type="password", key="session_token")
        
        if st.button("Update AWS Credentials Manually"):
            with st.spinner("Updating AWS credentials..."):
                if st.session_state.bot.refresh_aws_credentials(
                    access_key=access_key if access_key else None,
                    secret_key=secret_key if secret_key else None,
                    session_token=session_token if session_token else None
                ):
                    st.success("✅ AWS credentials updated successfully!")
                else:
                    st.error("❌ Failed to update AWS credentials")
    
    # Add database explorer
    if st.sidebar.button("Explore Databases"):
        st.sidebar.subheader("Database Explorer")
        try:
            with st.spinner("Fetching databases..."):
                databases = st.session_state.bot.athena_client.list_databases()
                db_names = [db['Name'] for db in databases['DatabaseList']]
                
                selected_db = st.sidebar.selectbox("Select Database", db_names, index=db_names.index("demo") if "demo" in db_names else 0)
                
                if selected_db:
                    tables = st.session_state.bot.athena_client.list_table_metadata(
                        CatalogName='AwsDataCatalog',
                        DatabaseName=selected_db
                    )
                    table_names = [table['Name'] for table in tables['TableMetadataList']]
                    
                    selected_table = st.sidebar.selectbox("Select Table", table_names)
                    
                    if selected_table and st.sidebar.button("Preview Table"):
                        with st.spinner(f"Fetching preview of {selected_db}.{selected_table}..."):
                            preview_df = st.session_state.bot.run_athena_query(f"SELECT * FROM {selected_db}.{selected_table} LIMIT 10", database=selected_db)
                            st.dataframe(preview_df)
                            
                            # Show row count
                            count_df = st.session_state.bot.run_athena_query(f"SELECT COUNT(*) as row_count FROM {selected_db}.{selected_table}", database=selected_db)
                            st.write(f"Total rows: {count_df['row_count'].iloc[0]}")
        except Exception as e:
            # Check if this is a token expiration error
            if "expired" in str(e).lower() or "token" in str(e).lower() or "credentials" in str(e).lower():
                st.sidebar.error("AWS credentials expired. Please refresh your token in the AWS Authentication section.")
                # Try automatic refresh
                with st.spinner("Attempting automatic OAuth refresh..."):
                    if st.session_state.bot.refresh_aws_credentials_oauth():
                        st.sidebar.success("✅ AWS credentials refreshed automatically!")
                        st.experimental_rerun()  # Rerun the app to reflect changes
                    else:
                        st.sidebar.error("❌ Automatic refresh failed. Please refresh manually.")
            else:
                st.sidebar.error(f"Error exploring databases: {str(e)}")
    
    if st.sidebar.button("Refresh Data"):
        with st.spinner("Refreshing data..."):
            st.session_state.bot.initialize(force_refresh=True)
        st.success("Data refreshed!")
    
    # Add data viewer section
    if st.sidebar.button("View Raw Data"):
        st.sidebar.subheader("Database Tables")
        try:
            with st.spinner("Fetching data..."):
                # Fetch contract data
                contracts_df = st.session_state.bot.run_athena_query("SELECT * FROM demo.files LIMIT 100", database="demo")
                contacts_df = st.session_state.bot.run_athena_query("SELECT * FROM demo.contact LIMIT 100", database="demo")
                
                # Display in expanders
                with st.expander("Contracts Data (files table)"):
                    st.dataframe(contracts_df)
                
                with st.expander("Contacts Data (contact table)"):
                    st.dataframe(contacts_df)
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["expired", "token", "credentials"]):
                st.error("AWS credentials expired. Please refresh your token in the AWS Authentication section.")
                # Try automatic refresh
                with st.spinner("Attempting automatic OAuth refresh..."):
                    if st.session_state.bot.refresh_aws_credentials_oauth():
                        st.success("✅ AWS credentials refreshed automatically!")
                        st.experimental_rerun()  # Rerun the app to reflect changes
                    else:
                        st.error("❌ Automatic refresh failed. Please refresh manually.")
            else:
                st.error(f"Error fetching data: {str(e)}")
    
    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if question := st.chat_input("Ask a question about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.bot.ask(question)
                    st.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = str(e)
                    
                    # Check if this is a token expiration error
                    if any(keyword in error_message.lower() for keyword in ["expired", "token", "credentials"]):
                        st.error("AWS credentials expired. Attempting automatic refresh...")
                        # Try automatic refresh
                        if st.session_state.bot.refresh_aws_credentials_oauth():
                            st.success("✅ AWS credentials refreshed automatically! Retrying your question...")
                            try:
                                response = st.session_state.bot.ask(question)
                                st.markdown(response)
                                # Add assistant response to chat history
                                st.session_state.messages.append({"role": "assistant", "content": response})
                            except Exception as retry_e:
                                st.error(f"Error after credential refresh: {str(retry_e)}")
                        else:
                            st.error("❌ Automatic refresh failed. Please refresh your token in the AWS Authentication section.")
                    else:
                        st.error(f"Error: {error_message}")

# Command-line interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Airbyte Data Chatbot")
    parser.add_argument('--webapp', action='store_true', help='Run the web app')
    parser.add_argument('--refresh', action='store_true', help='Force refresh of the vector store')
    parser.add_argument('--test', action='store_true', help='Run with sample data (no AWS needed)')
    parser.add_argument('--query', type=str, help='Run a direct Athena query and display results')
    parser.add_argument('--database', type=str, default="demo", help='Database to query (default: demo)')
    parser.add_argument('--diagnose', action='store_true', help='Run diagnostics on AWS connection')
    parser.add_argument('--discover', action='store_true', help='Discover available tables in AWS')
    parser.add_argument('--oauth-refresh', type=str, help='Refresh AWS credentials using OAuth refresh token', nargs='?', const='')
    args = parser.parse_args()
    
    # Run web app if requested
    if args.webapp:
        print("Starting Streamlit web application...")
        # Don't try to programmatically launch Streamlit
        print("Please run with: streamlit run langchain_efficient_demo.py")
        return
    
    # Command-line mode
    bot = AwsLangChainBot()
    
    # Process OAuth refresh if requested
    if args.oauth_refresh is not None:
        print("\n=== AWS OAuth Refresh ===")
        refresh_token = args.oauth_refresh if args.oauth_refresh else None
        
        if bot.refresh_aws_credentials_oauth(refresh_token):
            print("✅ AWS credentials refreshed successfully via OAuth!")
            return
        else:
            print("❌ Failed to refresh AWS credentials via OAuth")
            return
    
    # Run diagnostics if requested
    if args.diagnose:
        # Initialize AWS clients first
        try:
            bot.initialize_aws_clients()
        except Exception as e:
            # If credentials are expired, try OAuth refresh
            if "expired" in str(e).lower() or "token" in str(e).lower():
                print("⚠️ AWS credentials expired. Attempting OAuth refresh...")
                if bot.refresh_aws_credentials_oauth():
                    print("✅ Credentials refreshed automatically!")
                else:
                    print("❌ OAuth refresh failed. Running with sample data.")
        
        bot.run_diagnostics()
        return
    
    # Discover tables if requested
    if args.discover:
        # Initialize AWS clients first
        try:
            bot.initialize_aws_clients()
        except Exception as e:
            # If credentials are expired, try OAuth refresh
            if "expired" in str(e).lower() or "token" in str(e).lower():
                print("⚠️ AWS credentials expired. Attempting OAuth refresh...")
                if bot.refresh_aws_credentials_oauth():
                    print("✅ Credentials refreshed automatically!")
                else:
                    print("❌ OAuth refresh failed. Running with sample data.")
        
        bot.discover_tables()
        return
    
    # If direct query mode is requested
    if args.query:
        try:
            # Initialize AWS clients first
            try:
                bot.initialize_aws_clients()
            except Exception as e:
                # If credentials are expired, try OAuth refresh
                if "expired" in str(e).lower() or "token" in str(e).lower():
                    print("⚠️ AWS credentials expired. Attempting OAuth refresh...")
                    if bot.refresh_aws_credentials_oauth():
                        print("✅ Credentials refreshed automatically!")
                    else:
                        print("❌ OAuth refresh failed. Cannot run query.")
                        return
                else:
                    raise
            
            print(f"Running query: {args.query}")
            result_df = bot.run_athena_query(args.query, database=args.database)
            print("\nQuery Results:")
            print(result_df.to_string())
            return
        except Exception as e:
            print(f"Error running query: {str(e)}")
            return
    
    print("\n===== Airbyte Data Chatbot =====")
    
    # Test AWS connection if not explicitly in test mode
    if not args.test:
        try:
            bot.initialize_aws_clients()
            print("✅ AWS Connection: Active")
        except Exception as e:
            if "expired" in str(e).lower() or "token" in str(e).lower():
                print("⚠️ AWS token expired. Attempting OAuth refresh...")
                
                if bot.refresh_aws_credentials_oauth():
                    print("✅ AWS credentials refreshed automatically via OAuth!")
                else:
                    print("⚠️ OAuth refresh failed. Would you like to:")
                    print("1. Enter AWS credentials manually")
                    print("2. Run with sample data")
                    choice = input("Choose an option (1/2): ")
                    
                    if choice == '1':
                        session_token = input("Enter new AWS Session Token: ")
                        if bot.refresh_aws_credentials(session_token=session_token):
                            print("✅ AWS session token updated successfully!")
                        else:
                            print("❌ Failed to update token. Running with sample data instead.")
                            print("💡 Use --test flag to suppress this warning")
                    else:
                        print("Running with sample data instead.")
                        print("💡 Use --test flag to suppress this warning")
            else:
                print(f"❌ AWS Connection Failed: {str(e)}")
                print("⚠️ Running with sample data instead")
                print("💡 Use --test flag to suppress this warning")
    else:
        print("🧪 Running in test mode with sample data")
    
    print("Initializing (this may take a minute)...")
    
    if not bot.initialize(force_refresh=args.refresh):
        print("Failed to initialize. Exiting.")
        return
    
    print("\nAsk questions about contracts from Google Drive and contacts from Salesforce")
    print("\n++ 🐙Powered by Airbyte 🐙 ++")
    
    
    while True:
        question = input("\nQuestion: ")
        
        if question.lower() in ['exit', 'quit', 'q']:
            break
        elif question.lower() == 'refresh':
            print("Refreshing vector store...")
            bot.initialize(force_refresh=True)
            continue
        elif question.lower() == 'token':
            # Just update the session token (common for temporary credentials)
            print("\n--- Update AWS Session Token ---")
            session_token = input("Enter new AWS Session Token: ")
            
            if bot.refresh_aws_credentials(session_token=session_token):
                print("✅ AWS session token updated successfully!")
            continue
        elif question.lower() == 'oauth':
            # Refresh using OAuth
            print("\n--- Refresh AWS Credentials using OAuth ---")
            refresh_token = input("Enter AWS refresh token (leave empty to use stored token): ")
            
            if bot.refresh_aws_credentials_oauth(refresh_token if refresh_token else None):
                print("✅ AWS credentials refreshed successfully via OAuth!")
            else:
                print("❌ Failed to refresh AWS credentials via OAuth")
            continue
        elif question.lower().startswith('sql:'):
            # Direct SQL query mode
            sql_query = question[4:].strip()
            try:
                print(f"Running SQL query: {sql_query}")
                result_df = bot.run_athena_query(sql_query)
                print("\nQuery Results:")
                print(result_df.to_string())
            except Exception as e:
                print(f"Error running query: {str(e)}")
            continue
        elif question.lower() == 'explore':
            bot.explore_data()
            continue
        elif question.lower() == 'diagnose':
            print("Running AWS diagnostics...")
            bot.run_diagnostics()
            continue
        elif question.lower() == 'discover':
            print("Discovering available tables...")
            bot.discover_tables()
            continue
        
        try:
            answer = bot.ask(question)
            print(f"\nAnswer: {answer}")
        except Exception as e:
            if "expired" in str(e).lower() and "token" in str(e).lower():
                print("⚠️ AWS token expired during query. Attempting OAuth refresh...")
                if bot.refresh_aws_credentials_oauth():
                    print("✅ AWS credentials refreshed automatically via OAuth! Please try your question again.")
                    try:
                        # Retry the question with fresh credentials
                        answer = bot.ask(question)
                        print(f"\nAnswer: {answer}")
                    except Exception as retry_e:
                        print(f"Error after credential refresh: {str(retry_e)}")
                else:
                    print("❌ OAuth refresh failed.")
                    print("Would you like to update your AWS session token manually? (y/n)")
                    if input().lower() == 'y':
                        session_token = input("Enter new AWS Session Token: ")
                        if bot.refresh_aws_credentials(session_token=session_token):
                            print("✅ AWS session token updated successfully! Please try your question again.")
                        else:
                            print("❌ Failed to update token.")
            else:
                print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()