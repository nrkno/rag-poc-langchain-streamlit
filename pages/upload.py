import os
import streamlit as st

st.set_page_config(
    page_title="Upload",
    page_icon="📤",
)

uploaded_file = st.file_uploader("Upload a file")

if uploaded_file is not None:
    st.write("File uploaded successfully.")
    uploaded_file.type
    with open(f"data/{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getvalue())

st.write("Files in data folder:")
st.write(os.listdir("data/"))
