import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings(action = 'ignore')


@st.cache_resource
def load_data(data):
    df = pd.read_csv(data)
    return df

def run_eda_app():
    st.subheader("Exploratory Data Analysis")

    # df = pd.read_csv("C:/Users/Chesta/Downloads/playground-series-s3e11/train.csv")
    df = load_data('C:/Users/Chesta/Downloads/playground-series-s3e11/train.csv')
    

   
    submenu = st.sidebar.selectbox('Submenu',['Data Description','Plots'])
    if submenu=="Data Description":
        st.subheader("Data Description")
        st.dataframe(df)
        st.write(f"Shape of DataFrame is:- {df.shape}")
        with st.expander("Data Types"):
            st.dataframe(df.dtypes)
        with st.expander("Descriptive Summary"):
            st.dataframe(df.describe())
        
    elif submenu=='Plots':
        st.subheader('Plots')
        with st.expander("Boxplot & Countplot"):
            st.subheader("Boxplot and average costing of ordinal variables")
            grouped_data = df.groupby(['unit_sales(in millions)'])['cost'].agg('mean').reset_index()

            fig, axs = plt.subplots(nrows =4,ncols=2, figsize=(10, 7))
            garph = sns.boxplot(x='unit_sales(in millions)',y='cost',hue='unit_sales(in millions)',data=df,dodge=False,palette = 'winter',ax=axs[0][0])
            garph.get_legend().remove()
            # st.pyplot(plt)

            garph2 = sns.barplot(x='unit_sales(in millions)',y='cost',data=grouped_data,hue='unit_sales(in millions)',dodge=False,ax=axs[0][1],palette = 'winter')
            # plt.legend(bbox_to_anchor=(1.02, 1), loc='upper right', borderaxespad=0)
            garph2.get_legend().remove()
            plt.tight_layout()
            # st.pyplot(plt)


            garph3 = sns.boxplot(x='total_children',y='cost',hue='total_children',data=df,dodge=False,palette = 'winter',ax=axs[1][0])
            garph3.get_legend().remove()
            # st.pyplot(plt)

            grouped_data_cost = df.groupby(['total_children'])['cost'].agg('mean').reset_index()
            garph4 = sns.barplot(x='total_children',y='cost',data=grouped_data_cost,hue='total_children',dodge=False,ax=axs[1][1],palette = 'winter')
            garph4.get_legend().remove()
            plt.tight_layout()
            # st.pyplot(plt)


            garph3 = sns.boxplot(x='num_children_at_home',y='cost',hue='num_children_at_home',data=df,dodge=False,palette = 'winter',ax=axs[2][0])
            garph3.get_legend().remove()
            # st.pyplot(plt)

            grouped_data_cost = df.groupby(['num_children_at_home'])['cost'].agg('mean').reset_index()
            garph4 = sns.barplot(x='num_children_at_home',y='cost',data=grouped_data_cost,hue='num_children_at_home',dodge=False,ax=axs[2][1],palette = 'winter')
            garph4.get_legend().remove()
            plt.tight_layout()
            # st.pyplot(plt)


            garph3 = sns.boxplot(x='avg_cars_at home(approx).1',y='cost',hue='avg_cars_at home(approx).1',data=df,dodge=False,palette = 'winter',ax=axs[3][0])
            garph3.get_legend().remove()
            # st.pyplot(plt)

            grouped_data_cost = df.groupby(['avg_cars_at home(approx).1'])['cost'].agg('mean').reset_index()
            garph4 = sns.barplot(x='avg_cars_at home(approx).1',y='cost',data=grouped_data_cost,hue='avg_cars_at home(approx).1',dodge=False,ax=axs[3][1],palette = 'winter')
            garph4.get_legend().remove()
            plt.tight_layout()
            st.pyplot(plt)
        
        with st.expander("Boxplot and Distplot"):
            st.subheader("Boxplot and densityplot of binary variables in respect to costing")
            binary_data = ['florist','salad_bar','video_store','coffee_bar','recyclable_package','low_fat']
            i=1
            j=2

            plt.figure(figsize = [10,15])
            plt.subplots_adjust(hspace = 0.5)

            for col in binary_data:
            # print(col)
                plt.subplot(6,2,i)
                sns.boxplot(x=col,y='cost',data=df,palette='winter',hue=col)
                plt.tight_layout()

            
                plt.subplot(6,2,j)

                sns.kdeplot(data = df,
                            x = 'cost',
                            hue = col,
                            fill = 'stack',
                            palette = 'winter',
                            shade = True,
                            legend = False)
                
                
                plt.title(col)
                plt.tight_layout()
            
              

                i+=2
                j+=2
            st.pyplot(plt)
        with st.expander("Histogram plot"):
            st.subheader("Histogram plot of Continuous variables")
            numerical_col= ['cost','store_sales(in millions)','gross_weight','units_per_case','store_sqft']
            i = 1
            plt.figure(figsize = [10,15])
            plt.subplots_adjust(hspace = 0.5)
            for col in numerical_col:
                plt.subplot(5,1,i)
                sns.distplot(x=df[col],color='blue')
                plt.title(col)
                i+=1
                plt.tight_layout()
            st.pyplot(plt)

        
        
        with st.expander("Correlation Matrix"):
          
            st.write("# **Correlation Matrix Conclusions Train Data**")
            plt.figure(figsize=(15,10))
            correlation = df.corr()
            mask = np.triu(np.ones_like(correlation, dtype=bool))
            sns.heatmap(correlation,annot=True,mask=mask)
            plt.title('Correlation Matrix Among Variables of Train Data')
            st.pyplot(plt)

            st.write("While looking at the correlation matrix among variables of the data set we can make certain conclusions")
            st.write("* Unit sales in millions have a positive relation with the store sales in millions - 0.46. As unit sales or quantity of items sold in the stores increasing this leads to increase in  total sales of stores")
            st.write("* there is significant correlation among the facilities(florist,video_bar,coffee_bar,salad_bar) which have been avialbale by the stores that might impct the costing of the campaigns.")
            st.write("* Preparedfood and salad bar has significant correlation that can cause **multicollinearity** because of which we can remove one of the variable.")
            st.write("* Store_sq_ft has significant postive relation in respect to Salad_bar and prepared_food. With this we can make an assumption more focus is towards the salad_food by the stores.")
            st.write("**Multicollinearity** When features are highly correlated to one another. The easiest way to visualize multicollinearity is by creating the Correlation Matrix.")
            st.write("Over here we can see there is perfect correlation between **prepared_food** and **salad_bar** having value of 1.")
         
        with st.expander("Boxplot"):
            st.subheader("Unit sales in millions with respect to store sales")
            plt.figure(figsize=(15,10))
            sns.boxplot(x='unit_sales(in millions)',y='store_sales(in millions)',data=df,palette='winter')
            st.pyplot(plt)