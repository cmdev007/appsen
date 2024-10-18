import streamlit as st
from main import analyze_app_reviews
import threading
import time
import uuid
import queue

gemini_api = st.secrets["gemini_api"]

# Initialize session state variables
if 'summary' not in st.session_state:
    st.session_state.summary = None
if 'top_problems' not in st.session_state:
    st.session_state.top_problems = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'analysis_stopped' not in st.session_state:
    st.session_state.analysis_stopped = False

st.set_page_config(page_title="AppSen: App Review Analyzer", page_icon="📱")

st.title("📱 AppSen: App Review Analyzer")

st.markdown("""
Welcome to **AppSen**, an AI-powered tool to analyze app reviews and identify key issues users are experiencing.
""")

app_name = st.text_input("🔍 **Enter the app name:**")

# Create a flag to control the analysis process
stop_analysis = threading.Event()

# Initialize the queue
result_queue = queue.Queue()

def run_analysis():
    summary, top_problems = analyze_app_reviews(app_name, stop_analysis, gemini_api)
    if summary is not None and top_problems is not None:
        result_queue.put((summary, top_problems))
    else:
        result_queue.put(None)

if st.button("🚀 Analyze Reviews"):
    if app_name:
        stop_analysis.clear()
        st.session_state.analysis_complete = False
        st.session_state.analysis_stopped = False
        st.session_state.stop_button_key = str(uuid.uuid4())
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.start()

        # Define the stop button outside the loop
        stop_button = st.button("🛑 Stop Analysis", key=st.session_state.stop_button_key)

        with st.spinner("Analyzing reviews... This may take a few minutes."):
            while analysis_thread.is_alive():
                if stop_button:
                    stop_analysis.set()
                    st.warning("Stopping analysis...")
                    break
                time.sleep(0.1)

        # Check for results
        try:
            result = result_queue.get(block=False)
            if result:
                st.session_state.summary, st.session_state.top_problems = result
                st.session_state.analysis_complete = True
            else:
                st.session_state.analysis_stopped = True
        except queue.Empty:
            st.session_state.analysis_stopped = True

        if st.session_state.analysis_complete:
            st.subheader("📊 Analysis Results")
            # st.markdown("### 📝 Summary")
            # st.info(st.session_state.summary)
            st.markdown("### 🔝 Top Problems Identified")
            st.write(st.session_state.top_problems)
        elif st.session_state.analysis_stopped:
            st.warning("Analysis was stopped before completion.")
    else:
        st.warning("Please enter an app name.")

st.sidebar.markdown("## About")
st.sidebar.info("""
**AppSen** analyzes reviews for a given app and identifies the top problems using AI.

- 🔍 Enter the app name.
- 🚀 Click **Analyze Reviews** to start.
- 🛑 Use **Stop Analysis** to halt the process anytime.
""")