import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

header = st.beta_container()
dataset = st.beta_container()
eda = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()

@st.cache
def get_data(filename):
     customers = pd.read_csv(filename)
     return customers

with header:
    st.title("Linear Regression!")
    st.markdown(" An Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want. The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired you on contract to help them figure it out! Let's get started!")
    

with dataset:
    st.header("Ecommerce Company Dataset!")
    st.text("This dataset can be found on kaggle.com")
    
    customers = get_data("Ecommerce Customers")
    st.write(customers.head(20))


with eda:
     st.header("Exploratory Data Analysis!")
     st.text("")

     st.subheader("A plot to compare the Time on Website and Yearly Amount Spent.")
     fig,ax = plt.subplots(figsize=(12,6))
     ax.scatter(customers['Time on Website'],customers['Yearly Amount Spent'])
     ax.set(ylabel='Yearly Amount Spent',
     xlabel="Time on Website")
     st.pyplot(fig)

     st.subheader("A plot to compare the Time on App and Yearly Amount Spent.")

     fig,ax1 = plt.subplots(figsize=(12,6))
     ax1.scatter(customers['Time on App'],customers['Yearly Amount Spent'])
     ax1.set(ylabel='Yearly Amount Spent',
     xlabel="Time on App")
     st.pyplot(fig)

     st.subheader("A plot to compare the Time on App and Length of Membership.")

     fig,ax2 = plt.subplots(figsize=(12,6))
     ax2.scatter(customers['Time on App'],customers['Length of Membership'])
     ax2.set(ylabel='Length of Membership',
     xlabel="Time on App")
     st.pyplot(fig)

     st.subheader("A plot to compare the Yearly Amount Spent and Length of Membership.")

     fig,ax3 = plt.subplots(figsize=(12,6))
     ax3.scatter(customers['Yearly Amount Spent'],customers['Length of Membership'])
     ax3.set(ylabel='Length of Membership',
     xlabel="Yearly Amount Spent")
     st.pyplot(fig)

     st.subheader("Correlation of the features with Yearly Amount Spent")
     st.write(customers.corr())

    
    

     

with features:
     st.header("Features Required!")
     st.markdown("* Avg. Session Length: Average session of in-store style advice sessions.")
     st.markdown("* Time on App: Average time spent on App in minutes.")
     st.markdown("* Time on Website: Average time spent on Website in minutes.")
     st.markdown("* Length of Membership: How many years the customer has been a member.")


with model_training:
     st.header("Time to train the model!")
     st.text("Here you choose the hyperparameters of the model") 

     disp_col,sel_col = st.beta_columns(2)

     X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

     y = customers['Yearly Amount Spent']

     input_feature = st.sidebar.multiselect("Which feature do you want to use?",X.columns)
     

     X = customers[input_feature]

     
     X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=101) 


      

     lm = LinearRegression()
     lm.fit(X_train,y_train)
     prediction = lm.predict(X_test)


     disp_col.subheader("Mean absolute error:")
     disp_col.write(metrics.mean_absolute_error(y_test,prediction))

     disp_col.subheader("Mean squared error:")
     disp_col.write(metrics.mean_squared_error(y_test,prediction))

     disp_col.subheader(" Root Mean absolute error:")
     disp_col.write(np.sqrt(metrics.mean_squared_error(y_test,prediction)))

     st.subheader("Coefficients:")
     
     coeffecients = pd.DataFrame(lm.coef_,X.columns)
     coeffecients.columns = ['Coeffecient']
     st.write(coeffecients)



