import streamlit as st
import numpy as np

import pandas as pd

from sklearn.tree import DecisionTreeClassifier

# Importing the dataset
data_url='https://raw.githubusercontent.com/memudualimatou/THE-SPARKS-FOUNDATION-GRIP/master/Iris.csv'
data = pd.read_csv(data_url,sep=",")

st.markdown("<h1 style='text-align: center; color: green; margin-bottom=0'>Iris Classification Web Application<h1>",
            unsafe_allow_html=True)
st.markdown(">Welcome to This web app which aim is to categorize iris based on  imputed features.\n\n<br>",
            unsafe_allow_html=True)

st.image("Iris_versicolor_4.jpg", use_column_width=True)

st.markdown("<h1 style='text-align:center; color:black'>ENTER DATA INPUT<br>", unsafe_allow_html=True)


def user_input():
    sl = st.number_input("Enter the sepal Length :")
    sw = st.number_input("Enter the sepal width :")

    pl = st.number_input("Enter the petal Length :")
    pw = st.number_input("Enter the petal width :")
    data = {
        'SepalLengthCm': sl,
        'SepalWidthCm': sw,
        'PetalLengthCm': pl,
        'PetalWidthCm': pw,
    }
    features = pd.DataFrame(data, index=[0])
    return features


user_df = user_input()
st.markdown("<br> User Dataframe", unsafe_allow_html=True)
st.write(user_df)

X = np.array(data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
y = np.array(data['Species'])
model = DecisionTreeClassifier(criterion='entropy', random_state=0)
model.fit(X, y)
#
st.markdown("<br><br>The Iris Category is ", unsafe_allow_html=True)
st.write(model.predict(user_df))

st.markdown("<h1 style='text-align:center; color:green'><br><br>THANK YOU !", unsafe_allow_html=True)
