import streamlit as st
import requests
import json
import time

# =======================
# Configuration & Styles
# =======================
st.set_page_config(
    page_title="Universal PDF Extractor",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max_width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4CAF50; 
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 40px;
        text-align: center;
        background-color: white;
        margin-bottom: 20px;
    }
    .json-box {
        background-color: #282c34;
        color: #abb2bf;
        padding: 20px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        overflow-x: auto;
    }
    h1, h2, h3 {
        color: #333;
        font-family: 'Helvetica Neue', sans-serif;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# =======================
# Sidebar
# =======================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/337/337946.png", width=80)
    st.title("Settings")
    
    api_url = st.text_input("Backend API URL", value="http://127.0.0.1:8000/upload-pdf")
    
    st.markdown("---")
    st.markdown("""
    ### About
    This tool uses **Gemini AI** to extract structured JSON data from **ANY** PDF invoice or document.
    
    **Features:**
    - Universal Extraction
    - Structured JSON Output
    - Modern Interface
    
    Made with ❤️ using Streamlit & FastAPI.
    """)
    st.markdown("---")
    st.info("Ensure your backend is running on port 8000!")

# =======================
# Main Layout
# =======================
# Title with logo
title_col1, title_col2 = st.columns([0.1, 0.9])
with title_col1:
    st.image("recon_logo.png", width=100)
with title_col2:
    st.title("Reckon Universal PDF Data Extractor")

st.markdown("### `Upload your PDF to instantly convert it into structured JSON`")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### 1. Upload Document")
    uploaded_file = st.file_uploader("Drag & drop your PDF here", type=["pdf"])

    if uploaded_file:
        st.success(f"✅ File loaded: **{uploaded_file.name}**")
        
        # Preview (first page as image - optional/advanced, sticking to text/viewer for now or just generic info)
        st.info(f"Size: {uploaded_file.size / 1024:.2f} KB | Type: {uploaded_file.type}")
        
        if st.button("🚀 Process PDF"):
            if not api_url:
                st.error("Please specify the API URL in the sidebar.")
            else:
                with st.spinner("🤖 Analyzing document with AI... this may take a moment"):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                        start_time = time.time()
                        response = requests.post(api_url, files=files)
                        end_time = time.time()
                        
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state['result'] = data
                            st.session_state['duration'] = end_time - start_time
                            st.toast("Processing Complete!", icon="🎉")
                        else:
                            st.error(f"❌ API Error: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Connection Error: {str(e)}")

with col2:
    st.markdown("#### 2. Extraction Results")
    
    if 'result' in st.session_state:
        result = st.session_state['result']
        duration = st.session_state.get('duration', 0)
        
        st.markdown(f"⏱️ *Processed in {duration:.2f} seconds*")
        
        tab1, tab2 = st.tabs(["📊 JSON View", "📝 Raw Data"])
        
        with tab1:
            st.json(result.get("data", {}))
            
            # Download button
            json_str = json.dumps(result.get("data", {}), indent=4)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"extracted_{int(time.time())}.json",
                mime="application/json"
            )

        with tab2:
            st.code(json.dumps(result, indent=4), language='json')
            
    else:
        st.markdown("""
        <div style="text-align: center; color: #888; padding: 50px; border: 2px dashed #ddd; border-radius: 10px;">
            <h3>Waiting for input...</h3>
            <p>Upload a file and click 'Process' to see the magic happen.</p>
        </div>
        """, unsafe_allow_html=True)
