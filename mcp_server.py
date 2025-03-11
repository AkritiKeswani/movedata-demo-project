import os
import json
import sys
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP
from langchain_efficient_demo import AwsLangChainBot
import pandas as pd

# Initialize the FastMCP server
mcp = FastMCP(
    title="AWS Data Chatbot MCP Server",
    description="MCP Server for AWS Data Chatbot using fastmcp framework",
    version="1.0.0"
)

# Initialize the bot instance
bot = None

def get_bot():
    """Initialize the bot if it hasn't been initialized yet"""
    global bot
    if bot is None:
        bot = AwsLangChainBot()
        try:
            bot.initialize()
        except Exception as e:
            print(f"Warning: Bot initialization error: {e}")
    return bot

@mcp.tool()
async def ask_question(question: str) -> Dict[str, str]:
    """
    Ask a question about your Contract data - powered by Airbyte.
    
    Special commands:
    - "ask Airbyte: <sql_query> [in database: <database>]" - Ask AI bot questions about Contracts
   
    """
    bot = get_bot()
    
    # Check for special commands
    question_lower = question.lower()
    
    # Handle credential refresh
    if question_lower.startswith("refresh credentials:"):
        try:
            # Parse credentials from the question
            parts = question.split(":", 1)[1].strip().split()
            access_key = parts[0] if len(parts) > 0 else None
            secret_key = parts[1] if len(parts) > 1 else None
            session_token = parts[2] if len(parts) > 2 else None
            
            success = bot.refresh_aws_credentials(
                access_key=access_key,
                secret_key=secret_key,
                session_token=session_token
            )
            
            return {
                "answer": "AWS credentials refreshed successfully." if success else "Failed to refresh AWS credentials."
            }
        except Exception as e:
            return {"answer": f"Error refreshing credentials: {str(e)}"}
    
    # Handle OAuth refresh
    elif question_lower.startswith("oauth refresh:"):
        try:
            # Parse refresh token from the question
            refresh_token = question.split(":", 1)[1].strip() or None
            
            success = bot.refresh_aws_credentials_oauth(refresh_token)
            
            return {
                "answer": "AWS credentials refreshed successfully via OAuth." if success else "Failed to refresh AWS credentials via OAuth."
            }
        except Exception as e:
            return {"answer": f"Error refreshing credentials via OAuth: {str(e)}"}
    
    # Handle SQL query
    elif question_lower.startswith("run query:"):
        try:
            # Parse query and optional database
            query_text = question.split(":", 1)[1].strip()
            database = "demo"  # Default database
            
            # Check if database is specified
            if "in database:" in query_text.lower():
                query_parts = query_text.lower().split("in database:")
                query = query_parts[0].strip()
                database = query_parts[1].strip()
            else:
                query = query_text
            
            result_df = bot.run_athena_query(query, database=database)
            
            # Format the results as a markdown table
            if not result_df.empty:
                # Convert DataFrame to markdown table
                markdown_table = result_df.to_markdown(index=False)
                return {
                    "answer": f"Query results:\n\n{markdown_table}\n\n{len(result_df)} rows returned."
                }
            else:
                return {"answer": "Query executed successfully but returned no results."}
        except Exception as e:
            error_message = str(e)
            # Check if this is a token expiration error
            if any(keyword in error_message.lower() for keyword in ["expired", "token", "credentials"]):
                # Try automatic refresh
                if bot.refresh_aws_credentials_oauth():
                    try:
                        # Retry with refreshed credentials
                        query_text = question.split(":", 1)[1].strip()
                        database = "demo"  # Default database
                        
                        # Check if database is specified
                        if "in database:" in query_text.lower():
                            query_parts = query_text.lower().split("in database:")
                            query = query_parts[0].strip()
                            database = query_parts[1].strip()
                        else:
                            query = query_text
                        
                        result_df = bot.run_athena_query(query, database=database)
                        
                        # Format the results as a markdown table
                        if not result_df.empty:
                            # Convert DataFrame to markdown table
                            markdown_table = result_df.to_markdown(index=False)
                            return {
                                "answer": f"Credentials refreshed automatically. Query results:\n\n{markdown_table}\n\n{len(result_df)} rows returned."
                            }
                        else:
                            return {"answer": "Credentials refreshed automatically. Query executed successfully but returned no results."}
                    except Exception as retry_e:
                        return {"answer": f"Error after credential refresh: {str(retry_e)}"}
                else:
                    return {"answer": "AWS credentials expired and automatic refresh failed. Please refresh your credentials."}
            else:
                return {"answer": f"Error executing query: {error_message}"}
    
    # Handle initialization
    elif question_lower.startswith("initialize"):
        try:
            force_refresh = "force refresh" in question_lower
            success = bot.initialize(force_refresh=force_refresh)
            
            return {
                "answer": "Initialization successful." if success else "Initialization failed."
            }
        except Exception as e:
            return {"answer": f"Error during initialization: {str(e)}"}
    
    # Handle table discovery
    elif question_lower == "discover tables":
        try:
            # Capture the output of discover_tables
            import io
            
            # Redirect stdout to capture output
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            
            # Call the method
            bot.discover_tables()
            
            # Get the output and restore stdout
            output = new_stdout.getvalue()
            sys.stdout = old_stdout
            
            # Format the output nicely
            return {"answer": f"Available tables:\n\n```\n{output}\n```"}
        except Exception as e:
            return {"answer": f"Error discovering tables: {str(e)}"}
    
    # Regular question - use the standard ask method
    try:
        answer = bot.ask(question)
        return {"answer": answer}
    except Exception as e:
        error_message = str(e)
        # Check if this is a token expiration error
        if any(keyword in error_message.lower() for keyword in ["expired", "token", "credentials"]):
            # Try automatic refresh
            if bot.refresh_aws_credentials_oauth():
                # Retry with refreshed credentials
                try:
                    answer = bot.ask(question)
                    return {"answer": "Credentials refreshed automatically. " + answer}
                except Exception as retry_e:
                    return {"answer": f"Error after credential refresh: {str(retry_e)}"}
            else:
                return {"answer": "AWS credentials expired and automatic refresh failed. Please refresh your credentials."}
        else:
            return {"answer": f"Error: {error_message}"}

if __name__ == "__main__":
    # Run the MCP server
    mcp.run() 