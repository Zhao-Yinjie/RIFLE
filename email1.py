import streamlit as st
import numpy as np
from pandas import DataFrame
from keybert import KeyBERT
# For Flair (Keybert)
# from flair.embeddings import TransformerDocumentEmbeddings
import seaborn as sns
# For download buttons
from functionforDownloadButtons import download_button
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from openai_api import Classifier
from rumour_detection_twitter.detect import RumourDetection


st.set_page_config(
    page_title="Refining Internet Information with Few-shot LEarning (RIFLE)",
    page_icon="🎈",
    layout="centered",
)
def _max_width_():
    max_width_str = f"max-width: 2400px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )
_max_width_()
st.title("🎈Refining Internet Information with Few-shot LEarning (RIFLE)")
st.header("Hi! I'm your internet information assistant.\n Define your own preferred categories and give me a few examples!")
de, d1, de, d2, d3 = st.columns([0.07, 1, 0.07, 5, 0.07])
with d1:
    Category1 = st.text_input(label="Category1", placeholder="Academic")
    Category2 = st.text_input(label="Category2", placeholder="Events")
    Category3 = st.text_input(label="Category3", placeholder="Ads")
    Category4 = st.text_input(label="Category4", placeholder="School")
    Category5 = st.text_input(label="Category5", placeholder="Others")
with d2:
    Index1 = st.text_input(label="Index1", placeholder="Academic Email Index")
    Index2 = st.text_input(label="Index2", placeholder="Events Email Index")
    Index3 = st.text_input(label="Index3", placeholder="Ads Email Index")
    Index4 = st.text_input(label="Index4", placeholder="School Email Index")
    Index5 = st.text_input(label="Index5", placeholder="Others Email Index")

Category = [Category1, Category2, Category3, Category4, Category5]
Index = [Index1, Index2, Index3, Index4, Index5]

dictionary = dict(zip(Category, Index))
print(dictionary)


# ModelType = "Email (Default) 📧"
ModelType = st.radio(
            "Choose your dataset",
            ["Email (Default) 📧", "Youtube 📺"],
            help="At present, you can choose between email and youtube. More to come!",
            horizontal=True
        )

def addCate(Category,Index):
    # st.write(Category)
    # st.write(Index)
    # st.write(dictionary)
    classify = Classifier()
    file_type = ''
    if ModelType == "Email (Default) 📧":
        file_type = 'email'
    elif ModelType == "Youtube 📺":
        file_type = 'video'
    classify.run(mode= "few_shot", file_type= file_type, input_dict = dictionary)


if st.button('✨ Classify'):
    addCate(Category,Index)

def load_data():
    return pd.DataFrame(
        {
            "first column": [1, 2, 3, 4],
            "second column": [10, 20, 30, 40],
            "third column": [3, 20, 30, 40],
        }
    )

# Boolean to resize the dataframe, stored as a session state variable
st.checkbox("Use container width", value=False, key="use_container_width")

if ModelType == "Email (Default) 📧":
    # use model 1
    df = pd.read_csv('email_list.csv')

else:
    # use another model
    df = pd.read_csv('video_list.csv')

# df["Link"] = "https://streamlit.io/components?category=visualization"

st.dataframe(df, width=1300)

st.title("🎈AI Rumour Detection")

usr_input = st.text_input(label="Text to test", placeholder="Please enter the text you want to detect")

if st.button('✨ Is it a rumour?'):
    detector = RumourDetection()
    ans = detector.predict(usr_input)
    st.write(ans)
