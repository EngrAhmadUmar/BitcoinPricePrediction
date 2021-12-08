# %%writefile app.py%
import streamlit as st
import pickle
import openpyxl
import xlrd
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# loading the trained model
model = pickle.load(open('PickleModel.pkl','rb'))

#Importing numpy library and creating a function to split a univariate sequence into samples
from numpy import array
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


def main():
    # front end elements of the web page
    html_temp = """ 
    <div style ="background-color:#002E6D;padding:20px;font-weight:15px"> 
    <h1 style ="color:white;text-align:center;"> Bitcoin Prediction</h1> 
    </div> 
    """

    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html=True)
    default_value_goes_here = ""
    # ball_control = st.number_input("Please enter the players Ball Control Attribute", 0, 100000000, 0)
    # short_passing = st.number_input("Please enter the players Short Passing Attribute", 0, 100000000, 0)
    # dribbling = st.number_input("Please enter the players Dribbling Attribute", 0, 100000000, 0)
    # crossing = st.number_input("Please enter the players Crossing Attribute", 0, 100000000, 0)
    # curve = st.number_input("Please enter the players Curve Attribute", 0, 100000000, 0)

    uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")

    global dataframe
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        dataframe = df
    result = ""

    if st.button("Predict"):
      # Interpolating our data to fill up our N/A values
      dataframe = dataframe.interpolate()
      
      #Getting our features to be the Weighted_Price column
      test = dataframe['Weighted_Price']
          
      #Transforming the data into a 2D array and scalling it
      scaler = MinMaxScaler(feature_range=(0,1))
      test = scaler.fit_transform(np.array(test).reshape(-1,1))
      
      #Split into samples and features
      testx, testy = split_sequence(test, 100)
      
      #Reshape from [samples, timesteps] into [samples, timesteps, features]
      testx = testx.reshape((testx.shape[0], testx.shape[1], 1))
      
      #Running our prediction
      prediction = model.predict(testx)
     
      result = scaler.inverse_transform(prediction)
      
      #Displaying our prediction result
      st.write(result)


if __name__ == '__main__':
    main()
