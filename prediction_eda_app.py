import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit.components.v1 as stc
from PIL import Image
from eda_app import run_eda_app
from ml_app import run_ml_app

data = pd.read_csv('C:/Users/Chesta/Downloads/playground-series-s3e11/train.csv')
img = Image.open('C:/Users/Chesta/Pictures/kaggle.png')

def about():

    
    st.write('Our motto of this Streamlit application is to create a web app showing the basic EDA') 
    st.write('of the data set and the methodologies which have been applied for creating the model and helps in predicting the media-campaign cost.')

    st.write("1) We'll do EDA and provide our conclusions based on the analysis")
    st.write("2) Will cover some concepts based which we'll observe when working across the datasets like Multicollinearity, Importance of Scaling,Use of Cross Validation, Randomized Search.")
    st.write("3) We'll look for basic idea behind using ensemble learning technique. Specifically Random Forest which is using Bagging or Bootstrap Aggregation Advance Ensemble Technique.")


def main():
    st.image(img)
    st.title("Media-Campaign-Cost-Dataset Kaggle")
    menu = ['Home','EDA','ML','About']
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice=="Home":
        st.subheader('Home')
    elif choice=='EDA':
        run_eda_app()
    elif choice=='ML':
        run_ml_app()
    else:
        about()


if __name__ == '__main__':
    main()