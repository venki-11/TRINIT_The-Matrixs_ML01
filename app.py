import streamlit as st
import os
st.title("HelpBot")
st.write("""Select""")
chat=st.button("ChatBot",key=401)
news = st.button("Fake news predictor",key=101)
if chat:
    os.system("streamlit run chatrun.py")

elif news:
    os.system("streamlit run newsrun.py")

