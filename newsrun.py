import numpy as np
import pandas as pd
import itertools
import streamlit as st
from seqeval.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

#Read the data
df=pd.read_csv('newsdata.csv')
#Get shape and head
labels=df.label
#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['content'], labels, test_size=0.2, random_state=7)
#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)
clf=svm.SVC(kernel='linear')
clf.fit(tfidf_train,y_train)

def get_text1():
    inp=st.text_area("enter the news:")
    return inp
def fakepred():
    st.title("Fake news predictor")
    user_input = get_text1()
    ok1 = st.button("check",key=201)
    if ok1:
        uinp=[user_input]
        tfidf_inp = tfidf_vectorizer.transform(uinp)
        output = clf.predict(tfidf_inp)
        if output=="F":
            st.write("Fake news")
        else:
            st.write("Authentic data")
fakepred()