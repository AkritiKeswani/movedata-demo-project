import streamlit as st
from langchain_efficient_demo import AwsLangChainBot

def main():
    st.set_page_config(
        page_title="AWS Data Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AWS Data Chatbot")
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
    
    # Add database explorer
    if st.sidebar.button("Explore Databases"):
        st.sidebar.subheader("Database Explorer")
        try:
            with st.spinner("Fetching databases..."):
                databases = st.session_state.bot.athena_client.list_databases()
                db_names = [db['Name'] for db in databases['DatabaseList']]
                
                selected_db = st.sidebar.selectbox("Select Database", db_names)
                
                if selected_db:
                    tables = st.session_state.bot.athena_client.list_table_metadata(
                        CatalogName='AwsDataCatalog',
                        DatabaseName=selected_db
                    )
                    table_names = [table['Name'] for table in tables['TableMetadataList']]
                    
                    selected_table = st.sidebar.selectbox("Select Table", table_names)
                    
                    if selected_table and st.sidebar.button("Preview Table"):
                        with st.spinner(f"Fetching preview of {selected_db}.{selected_table}..."):
                            preview_df = st.session_state.bot.run_athena_query(
                                f"SELECT * FROM {selected_db}.{selected_table} LIMIT 10",
                                database=selected_db
                            )
                            st.dataframe(preview_df)
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
    
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
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 