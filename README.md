# movedata-demo-project

AWS Data Chatbot for querying Google Drive and Salesforce data in an S3 data lake.

## Overview

This project provides a chatbot interface to query and analyze data from Google Drive and Salesforce that has been stored in an AWS S3 data lake. The chatbot uses LangChain and AWS Athena to provide natural language querying capabilities.

## Running as an MCP Server for Claude Desktop

This project can be run as a Model Context Protocol (MCP) server, which allows it to be used directly from Claude Desktop.

### Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up your environment variables in a `.env` file:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_SESSION_TOKEN=your_session_token
   AWS_REGION=your_region
   OPENAI_API_KEY=your_openai_key
   ```

3. Run the MCP server:
   ```bash
   python mcp_server.py
   ```

4. In Claude Desktop, load the configuration:
   - Open Claude Desktop
   - Go to Settings > Developer
   - Click "Load Configuration"
   - Select the `claude_desktop_config.json` file

### Using the Chatbot

The MCP server provides a single unified interface through the `ask_question` tool. You can use this tool to:

1. **Ask natural language questions about your data:**
   ```
   Can you tell me about the contracts with the highest value?
   ```

2. **Perform administrative tasks using special commands:**

   - Refresh AWS credentials manually:
     ```
     refresh credentials: <access_key> <secret_key> <session_token>
     ```

   - Refresh AWS credentials using OAuth:
     ```
     oauth refresh: <refresh_token>
     ```

   - Run a direct SQL query:
     ```
     run query: SELECT * FROM demo.files LIMIT 10
     ```
     
     Or specify a different database:
     ```
     run query: SELECT * FROM contact LIMIT 10 in database: demo
     ```

   - Initialize or refresh the chatbot:
     ```
     initialize
     ```
     
     Force refresh the vector store:
     ```
     initialize force refresh
     ```

   - Discover available tables:
     ```
     discover tables
     ```

## Original Application

The original application can still be run in the following ways:

### Command Line Interface

```bash
python langchain_efficient_demo.py
```

### Streamlit Web Interface

```bash
streamlit run app.py
```

## Troubleshooting

If you encounter AWS credential issues, you can:

1. Use the special commands to refresh credentials:
   ```
   refresh credentials: <access_key> <secret_key> <session_token>
   ```
   or
   ```
   oauth refresh: <refresh_token>
   ```

2. Update your `.env` file with new credentials

3. Run the initialization again:
   ```
   initialize
   ```

## Notes for Developers

The MCP server implementation uses the fastmcp library, which has a simpler decorator syntax than what's shown in some MCP examples. Instead of using `input_schema` and `output_schema` parameters, we use Python type hints and docstrings to describe the tool's functionality.
