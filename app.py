
#  python -m streamlit run C:\Users\User\Downloads\TopicDetectorInDebate\mainTdetect.py


# https://plotly.com/python/discrete-color/
# https://github.com/MaartenGr/BERTopic

# imports
import streamlit as st
from PIL import Image
from collections import Counter
import pandas as pd
pd.set_option("max_colwidth", 400)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly
import plotly.graph_objects as go
import wordcloud
from wordcloud import WordCloud, STOPWORDS
import random
import re

from io import StringIO

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired

# Use a pipeline as a high-level helper
from transformers import pipeline

@st.cache_resource
def load_model_nli(model = "cross-encoder/nli-deberta-base"):
    pipe = pipeline("zero-shot-classification", model=model)
    return pipe



def add_spacelines(number_sp=2):
    for xx in range(number_sp):
        st.write("\n")


def split_into_para(text):
  colon_id = text.index(':') + 1
  text = text[colon_id: ].strip()
  exmpl = text.split(". ")
  nn = len(exmpl)

  splited = []

  if nn % 5 == 0:
    n0 = 0
    n1 = 5

    for _ in range( int(nn/5) ):
      #print( exmpl[n0:n1] )
      para_split = exmpl[n0:n1]
      para_split = ". ".join(para_split)
      splited.append( para_split )
      n0+=5
      n1+=5

  else:
    n0 = 0
    n1 = 6

    for _ in range( int( np.ceil(nn/6)) ):
      #print( exmpl[n0:n1] )
      para_split = exmpl[n0:n1]
      para_split = ". ".join(para_split)
      splited.append( para_split )
      n0+=6
      n1+=6
  return splited



def read_file(file):
    with tempfile.NamedTemporaryFile(mode="wb") as temp:
        bytes_data = file.getvalue()
        temp.write(bytes_data)
        df = MyFunctionReadsFromPathAndAggregations(temp.name)
        return df


#@st.cache_resource
def DetectT(docs, model, classes):
    topics, probs = model.fit_transform(docs)
    topics_per_class = model.topics_per_class(docs, classes=classes)  
    return topics, probs, topics_per_class, model
    

@st.cache_data
def SpkrInTime(data, chosen_categories):
    df_2 = data.copy().drop_duplicates('full_text_id')
    df_2 = df_2[ df_2.speaker_category.isin(chosen_categories) ]
    df_2 = df_2.reset_index(drop=True)
    df_2 = df_2.reset_index()    
    df_2["turn"] = df_2['index'].astype('int')
    df_2["velocity"] = 1
    return df_2




#  *********************** sidebar  *********************
with st.sidebar:
    #standard
    st.title("Navigator")
    add_spacelines(2)
    #st.write('Upload your debate in a **.txt** format')
    st.write( '**Data**' )  
    uploaded_file = st.file_uploader('', type = 'txt', accept_multiple_files = False, label_visibility = 'collapsed')
    if uploaded_file is not None:
        st.success('File uploaded correctly!')
    else:
        st.stop()
        
    add_spacelines(2)
    #st.write('Upload your debate in a **.txt** format')
    st.write( '**Parameters of topic detector**' ) 
    max_doc_freq = st.slider("Max doc frequency", 0.0, 1.0, 0.7)
    min_doc_freq = st.slider("Min doc frequency", 0, 30, 1)
    min_tsize = st.slider("Min topic size", 0, 50, 10)
    zero_shot_check = st.checkbox('Zero-shot topic detection')
    if zero_shot_check:
        zero_shot_check_list = st.text_area("Insert topic names, separated by a comma", value = 'war, abortion, immigration, tax economy, health care')
        zero_shot_check_list = zero_shot_check_list.split(", ")

        zeroshot_min_sim = st.slider("The minimum similarity between a zero-shot topic and a document for assignment", 0.0, 1.0, 0.35)
    else:
        zero_shot_check_list = None
        zeroshot_min_sim = 0.7
        
    

st.title('Topic detection in structured debates')
#with open(uploaded_file.name, 'r') as file:
#    data = uploaded_file.read()

data = uploaded_file.getvalue()
#st.write(data)

data_raw=StringIO(uploaded_file.getvalue().decode('utf-8'))
data_raw=data_raw.read()

data_list = data_raw.split("\r\n\r\n")

data = pd.DataFrame( {'text':data_list} )
data['speaker'] = data.text.apply(lambda x: x.split(":")[0] )
dd0 = data.speaker.value_counts().reset_index()
dd0 = dd0[dd0['count'] >= 5]
data = data[data.speaker.isin(dd0.speaker.unique())]

data['nwords'] = data.text.apply(lambda x: len( x.split() ) )
dd1 = data.groupby('speaker')['nwords'].sum().sort_values().reset_index()
dd1_politicians = dd1[dd1['nwords'] >= dd1['nwords'].mean()]['speaker'].tolist()
data['speaker_category'] = np.where( data.speaker.isin(dd1_politicians), 'politician', 'moderator' )

data['sentence'] = data.text.apply( lambda x: split_into_para(x) )

data2 = data.copy()
data2 = data2.explode('sentence')
data2 = data2.reset_index()

tab_topic_sum, tab_topic_vis, tab_topic_speakers, tab_df, tab_deb = st.tabs( ['Topics Summary', 'Topics Visualisation', 'Speakers', 'Dataframe', 'Debate' ] )

with tab_deb:
    st.write(data_raw)

#pipe_deberta = load_model_nli()

# Fine-tune your topic representations
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(ngram_range=(1, 1), stop_words="english", max_df = max_doc_freq, min_df = int(min_doc_freq) )
representation_model = KeyBERTInspired()
topic_model = BERTopic(representation_model=representation_model, min_topic_size = min_tsize, 
                       vectorizer_model=vectorizer_model, zeroshot_topic_list = zero_shot_check_list, zeroshot_min_similarity = zeroshot_min_sim)

docs = data2['sentence']
classes = data2[ 'speaker' ].tolist()
topics, probs = topic_model.fit_transform(docs)
topics_per_class = topic_model.topics_per_class(docs, classes=classes)  
#topics, probs, topics_per_class, topic_model = DetectT(docs = docs, model = topic_model, classes = classes)
#topics, probs, topics_per_class = DetectT(docs = docs, model = topic_model, classes = classes)
freq = topic_model.get_topic_info()



    
with tab_df:
    data3 = topic_model.get_document_info(docs)
    data3 = data3.rename(columns = {'Document':'sentence'})
    data2 = data3.merge(data2, on = 'sentence')
    data2 = data2.drop_duplicates('sentence')
    data2 = data2.rename(columns = {'index':'full_text_id'})
    st.write(data2)


with tab_topic_vis:
    fig = topic_model.visualize_topics( width = 750, height = 650 ) 
    st.plotly_chart( fig )
    add_spacelines(2)
    fig = topic_model.visualize_documents(docs, width = 850, height = 700 )
    st.plotly_chart( fig )
    add_spacelines(2)
    fig = topic_model.visualize_barchart(n_words = 8, width = 650, height = 600 )
    st.plotly_chart( fig )
    add_spacelines(2)

with tab_topic_speakers:        
    #topics_per_class = topic_model.topics_per_class(docs, classes=classes)    

    radio_value = st.radio( "Choose the unit for analysis", ['number', 'percentage'] )
    if radio_value == 'number':
        radio_value_bool = False
    else:
        radio_value_bool = True
    fig = topic_model.visualize_topics_per_class(topics_per_class, width = 750, height = 600, normalize_frequency = radio_value_bool )    
    st.plotly_chart( fig )
    add_spacelines(2)

    speaker_categories = list(data2.speaker_category.unique())  
    chosen_categories_in_time = st.multiselect( "Choose speaker categories to display", speaker_categories, speaker_categories )
    df_2 = SpkrInTime(data2, chosen_categories_in_time)

    fig = px.bar(df_2, x="turn", y="velocity", hover_data = {"speaker":True, "velocity":False}, color = "speaker",
                 labels={'velocity':'', 'speaker': 'Speaker'}, title = "Speakers distribution in time",
                 width=1000, height=460, color_discrete_sequence=px.colors.qualitative.G10)
    fig.update_layout(xaxis={"tickformat":"d"},
                      font=dict(size=15,color='#000000'),
                       yaxis = dict(tickmode = 'linear',tick0 = 0,dtick = 1
        ))
    
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig)

    
with tab_topic_sum:
    st.write(freq)
    add_spacelines(2)
    st.write(data2[ data2['Topic'] != -1 ]['Name'].value_counts(normalize=True).round(2)*100)
# https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html#visualize-topics-over-time
