import streamlit as st
import requests


st.title('Welcome to the Chatbot')

# Input field for user query
input_text = st.text_input('Ask me anything')


if input_text:
    with st.spinner("Processing..."):
        try:
            # API URL for sending user input
            api_url = "http://localhost:7001/chat"
            
            # Prepare the request payload in the required format
            payload = {"question": input_text}
            
            # Set the headers to specify content type
            headers = {'Content-Type': 'application/json'}
            
            # Send POST request to the API
            response = requests.post(api_url, json=payload, headers=headers)
            
            # Check if the response is successful
            if response.status_code == 200:
                # Parse the JSON response
                response_data = response.json()
                
                # Extract the 'description' from the 'documents' field if available
                description = response_data.get("documents", {}).get("description")
                
                # If 'description' is not found, fall back to 'page_content'
                if not description:
                    description = response_data.get("documents", {}).get("page_content", "No content found in response.")
                
                # Display the final generated description or content
                st.write(f"Response: {description}")
            else:
                # Display detailed error information
                st.error(f"Error: Received a {response.status_code} status code from the API.")
                st.write("Response Content:", response.text)
        
        except requests.exceptions.RequestException as e:
            # Display any request errors
            st.error(f"Request Error: {e}")
