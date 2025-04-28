import streamlit as st
import json

# Load knowledge base
@st.cache_data
def load_knowledge_base():
    with open("errors.json", "r") as file:
        return json.load(file)

kb = load_knowledge_base()
error_codes = [entry["error"] for entry in kb]

# App layout
st.title("Machine Fault Diagnostic Tool")

st.write("Select an error code or type it manually:")

# Option to select or type
selected_code = st.selectbox("Select error code:", options=error_codes)
manual_input = st.text_input("Or type error code manually (optional):").upper()

# Decide which input to use
search_code = manual_input if manual_input else selected_code

# Search function
def find_error_entry(code):
    for entry in kb:
        if entry["error"] == code:
            return entry
    return None

# Show result
if search_code:
    entry = find_error_entry(search_code)
    if entry:
        st.success(f"Match found for error code {search_code}")
        st.markdown(f"**Problem:** {entry['problem']}")
        st.markdown(f"**Cause:** {entry['causes']}")
        st.markdown(f"**Solution:** {entry['solution']}")
    else:
        st.warning("No matching entry found.")