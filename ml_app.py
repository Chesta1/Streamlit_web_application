import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings(action = 'ignore')

import pickle
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder

# model_load_in = open("C:/Users/Chesta/Downloads/model.pkl",'rb')
# model = load(model_load_in)
with open("C:/Users/Chesta/Desktop/Streamlit_app_folder/model1.pkl",'rb') as f:
    model_load = pickle.load(f)

def prediction(store_sales,gross_weight, units_per_case, total_children, num_children_at_home, store_sqft,unit_sales,avg_cars_at_home,recyclable_package,low_fat,coffee_bar,video_store,salad_bar,florist):
    input = pd.DataFrame([[store_sales,gross_weight, units_per_case, total_children, num_children_at_home, store_sqft,unit_sales,avg_cars_at_home,recyclable_package,low_fat,coffee_bar,video_store,salad_bar,florist]])
    #renaming of column
    input.columns = ['store_sales','gross_weight','units_per_case','total_children','num_children_at_home','store_sqft','unit_sales','avg_cars_at_home','recyclable_package','low_fat','coffee_bar','video_store','salad_bar','florist']
    input.rename(columns={'store_sales':'store_sales(in millions)','unit_sales':'unit_sales(in millions)','avg_cars_at_home':'avg_cars_at home(approx).1'},inplace=True)
    df = input
    st.write((df))
   
    # preprocessing_in = open('C:/Users/Chesta/Downloads/Preprocessorct_pipeline.pkl','rb')
    with open('C:/Users/Chesta/Desktop/Streamlit_app_folder/column_transformer_pipeline.pkl', 'rb') as f:
        ct_loaded = pickle.load(f)
    # st.write(preprocessing)
    # st.write(ct_loaded)
    input = ct_loaded.transform(input)
    # st.write((input))
    
    encoded_cols = ct_loaded.named_transformers_['onehot'].get_feature_names_out()
    # st.write(encoded_cols)
    num_cols = ct_loaded.named_transformers_['num'].get_feature_names_out()
    # st.write(num_cols)
    features = list(num_cols)+list(encoded_cols)
    input = pd.DataFrame(input,columns=features)
    st.write(input)

    prediction= model_load.predict(input)
    
    st.write(f"The Cost woulb be:- {round(prediction[0],2)}")
    # st.write(prediction[0])
def run_ml_app():
    st.subheader('Predictions for Costing of Campaign using Machine learning')
    """
    
store_sales(in millions)	float64
unit_sales(in millions)	float64
total_children	float64
num_children_at_home	float64
avg_cars_at home(approx).1	float64
gross_weight	float64
recyclable_package	float64
low_fat	float64
units_per_case	float64
store_sqft	float64
coffee_bar	float64
video_store	float64
salad_bar	float64
prepared_food	float64
florist	float64
cost	float64
    
    
    """

    col1,col2 = st.columns(2)
    # Pipeline
# preprocessor: ColumnTransformer
# num
# ['store_sales(in millions)', 'gross_weight', 'units_per_case', 'total_children', 'num_children_at_home', 'store_sqft']
# ['unit_sales(in millions)', 'avg_cars_at home(approx).1', 'recyclable_package', 'low_fat', 'coffee_bar', 'video_store', 'salad_bar', 'florist']
    with col1:
        store_sales = float(st.number_input("store_sales"))
        gross_weight = float(st.number_input("gross_weight"))
        units_per_case = float(st.number_input("units_per_case"))
        total_children = int(st.number_input("total_children",0,5))
        num_children_at_home = int(st.number_input("num_children_at_home",0,5))
        store_sqft = float(st.number_input("store_sqft"))
        unit_sales = int(st.number_input('unit_sales',1,6))
    with col2:
        avg_cars_at_home = int(st.number_input("average_cars",0,4))
        recyclable_package = int(st.number_input("recycable_pkg",0,1))
        low_fat = int(st.number_input("low_fat",0,1))
        coffee_bar =int(st.number_input('coffee_bar',0,1))
        video_store= st.number_input('video_store',0,1)
        salad_bar = st.number_input("salad_bar",0,1)
        florist=st.number_input("florist",0,1) 

    if st.button("Analysis Result"):
        analysis = prediction(store_sales,gross_weight, units_per_case, total_children, num_children_at_home, store_sqft,unit_sales,avg_cars_at_home,recyclable_package,low_fat,coffee_bar,video_store,salad_bar,florist)
        # st.success(analysis)   
    else:
        st.write("Click the above button for results")
  