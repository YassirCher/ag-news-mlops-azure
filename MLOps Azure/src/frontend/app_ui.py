import streamlit as st
import requests
import json

# Your Deployment URL (Keep the /predict/batch endpoint in mind or use single)
# REPLACE THIS with your actual Azure URL from the previous step
API_URL = "http://agnews-api-2912.spaincentral.azurecontainer.io:8000/predict"

st.set_page_config(page_title="AG News Classifier", page_icon="📰")

st.title("📰 AI News Classifier")
st.markdown("Enter a news headline below, and the AI will categorize it.")

# Input area
news_text = st.text_area("News Text", height=150, placeholder="e.g., Apple stock reaches all-time high...")

if st.button("Classify"):
    if news_text:
        try:
            with st.spinner("Analyzing..."):
                payload = {"text": news_text}
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    category = result['category']
                    confidence = result['confidence']
                    
                    # Color coding
                    color_map = {
                        "Business": "blue",
                        "Sci/Tech": "green",
                        "Sports": "orange",
                        "World": "red"
                    }
                    color = color_map.get(category, "grey")
                    
                    st.success(f"**Category:** :{color}[{category}]")
                    st.metric("Confidence", f"{confidence:.2%}")
                    
                    with st.expander("See details"):
                        st.json(result)
                else:
                    st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")
    else:
        st.warning("Please enter some text first.")
