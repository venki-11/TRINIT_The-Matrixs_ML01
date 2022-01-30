import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_text
import re
import os

#load data set
traindata = pd.read_excel("coviddata.xlsx")
pd.set_option('max_colwidth', 1000)

def preprocess_sentences(input_sentences):
    return [re.sub(r'(covid-19|covid)', 'coronavirus', input_sentence, flags=re.I)
            for input_sentence in input_sentences]

#load model
module = tf.saved_model.load(os.path.join("usemodel"))
response_encodings = module.signatures['response_encoder'](
        input=tf.constant(preprocess_sentences(traindata.Answer)),
        context=tf.constant(preprocess_sentences(traindata.Context)))['outputs']
i=1
def get_text():
    global i
    inp=st.text_area("you:",key=i)
    i+=1
    return inp
def pred():
    st.title("ChatBot")
    st.write("""Ask any queries related to covid and vaccination""")
    user_input = get_text()
    ok = st.button("send",key=i-1)
    if ok:
        query=[user_input]
        question_encodings = module.signatures['question_encoder'](
            tf.constant(preprocess_sentences(query))
            )['outputs']
        response = traindata.Answer[np.argmax(np.inner(question_encodings, response_encodings), axis=1)]
        out=response.to_string()
        st.markdown(out[2:])
pred()