import streamlit as st
import os
import numpy as np
import  pandas as pd
import seaborn as sns
import joblib

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

gender1= {'Male':1,'Female':0}
Partner_l = {'Yes':1 , 'No':0}
Dependents_l = {'Yes':1 , 'No':0}
PhoneService_l = {'Yes':1 , 'No':0}
OnlineSecurity_l = {'Yes':2 , 'No':1, 'No internet service':0}
OnlineBackup_l = {'Yes':2 , 'No':1 , 'No internet service':0}
DeviceProtection_l = {'Yes':2 , 'No':1 , 'No internet service':0}
TechSupport_l = {'Yes':2 , 'No':1 , 'No internet service':0}
StreamingMovies_l = {'Yes':2 , 'No':1 , 'No internet service':0}
StreamingTV_l = {'Yes':2 , 'No':1 , 'No internet service':0}
PaperlessBilling_l = {'Yes':1 , 'No':0}
InternetService_l = {'Fiber optic':2 , 'DSL':1, 'No':0}
MultipleLines_l = {'No phone service':0 , 'No':1 ,'Yes':2}
churn_l = {'Y':1 , 'N':0}
Contract_l = {'Month-to-month':0,
                                'One year':1,
                                'Two year':2,
                                }

PaymentMethod_l = {
'Electronic check'         :1,    
'Mailed check'             :2,   
'Bank transfer (automatic)':3,    
'Credit card (automatic)' :4
}


def get_value(val,my_dict):
	for key ,value in my_dict.items():
		if val == key:
			return value
        
def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key


def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


@st.cache
def load_data(dataset):
    df = pd.read_csv(dataset)
    return df


def main():
    ''' 
     Churn Analysis Machine Learning Prediction Modeling
    '''
    st.title('Churn Analysis Machine Learning App')
    st.subheader('5 Algorithm MachineLearning with Streamlit')

    menu = ['EDA','Prediction','About']
    choices = st.sidebar.selectbox('Select Activities' ,menu)

    if choices == 'EDA':
        st.subheader('EDA')

        data = load_data('/home/fingolfin/PycharmProjects/pythonProject/3ML/churn.csv')
        st.dataframe(data.head(10))

        if st.checkbox('Show Summary'):
            st.write(data.describe())
        if st.checkbox('Simple Value Plot'):
            st.write(sns.countplot(data['Churn']))
            st.pyplot()
        if st.checkbox('Select Columns to Show:'):
            all_columns =data.columns.tolist()
            selected_columns =st.multiselect('Select',all_columns)
            new_df = data[selected_columns]
            st.dataframe(new_df)
        if st.checkbox("Generate Pie Plot"):
            st.write(data.iloc[:, -1].value_counts().plot.pie(autopct="%1.1f%%"))
            st.pyplot()




    if choices == 'Prediction':
        st.subheader('Prediction')

        gender     = st.selectbox('Gender',tuple(gender1.keys()))
        Partner    = st.selectbox('Partner',tuple(Partner_l.keys()))
        Dependents = st.selectbox('Dependents',tuple(Dependents_l.keys()))
        PhoneService = st.selectbox('PhoneService',tuple(PhoneService_l.keys()))
        MultipleLines =st.selectbox('MultipleLines',tuple(MultipleLines_l.keys()))
        InternetService =st.selectbox('InternetService',tuple(InternetService_l.keys()))
        OnlineSecurity =st.selectbox('OnlineSecurity',tuple(OnlineSecurity_l.keys()))
        OnlineBackup =st.selectbox('OnlineBackUp',tuple(OnlineBackup_l.keys()))
        DeviceProtection =st.selectbox('DeviceProtection',tuple(DeviceProtection_l.keys()))
        TechSupport =st.selectbox('TechSupport',tuple(TechSupport_l.keys()))
        StreamingMovies =st.selectbox('StreamingMovies',tuple(StreamingMovies_l.keys()))
        Contract =st.selectbox('Contract',tuple(Contract_l.keys()))
        PaperlessBilling =st.selectbox('PaperlessBilling',tuple(PaperlessBilling_l.keys()))
        PaymentMethod =st.selectbox('PaymentMethod',tuple(PaymentMethod_l.keys()))
        StreamingTV =st.selectbox('StreamingTV',tuple(StreamingTV_l.keys()))
        seniorcitizen =st.number_input('SeniorCitizen',0,1)
        tenure        =st.number_input('Tenure',0,72)
        monthlycharges =st.number_input('MonthlyCharges',18,118)

        feature_list = [
             get_value(gender, gender1),
             get_value(Partner, Partner_l),
             get_value(Dependents, Dependents_l),
             get_value(PhoneService, PhoneService_l),
             get_value(MultipleLines, MultipleLines_l),
             get_value(InternetService, InternetService_l),
             get_value(OnlineSecurity, OnlineSecurity_l),
             get_value(OnlineBackup, OnlineBackup_l),
             get_value(DeviceProtection, DeviceProtection_l),
             get_value(TechSupport, TechSupport_l),
             get_value(StreamingMovies, StreamingMovies_l),
             get_value(Contract, Contract_l),
             get_value(PaperlessBilling, PaperlessBilling_l),
             get_value(PaymentMethod, PaymentMethod_l),
             get_value(StreamingTV, StreamingTV_l),
             seniorcitizen,tenure, monthlycharges

                           ]
        st.write(feature_list)
        pretty_data ={
        'gender'                :gender,
        'Partner'               :Partner,
        'Dependents'            : Dependents,
        'PhoneService'          : PhoneService,
        'MultipleLines'         :MultipleLines,
        'InternetService'       :InternetService,
        'OnlineSecurity'        :OnlineSecurity,
        'OnlineBackup'          :OnlineBackup,
        'DeviceProtection'      :DeviceProtection,
        'TechSupport'           :TechSupport,
        'StreamingMovies'       :StreamingMovies,
        'Contract'              :Contract,
        'PaperlessBilling'      :PaperlessBilling,
        'PaymentMethod'         :PaymentMethod,
        'StreamingTV'           :StreamingTV,
        'SeniorCitizen'         :seniorcitizen,
        'tenure'                :tenure,
        'MonthlyCharges'        :monthlycharges,
        }

        st.subheader('Option Selected')
        st.json(pretty_data)

        st.subheader('Data Encoded')

        prep_data = np.array(feature_list).reshape(1, -1)

        model_choice = st.selectbox('Model Choice',['RandomForest',
                                                    'DecisionTree',
                                                    'KNN',
                                                    'Naive Bayes',
                                                    'Logistic Regression'])
        if st.button('Evaluate'):
            if model_choice == 'RandomForest':
                predictor =load_prediction_models('RF5_loan.pkl')
                prediction = predictor.predict(prep_data)
                st.write(prediction)
            if model_choice == 'DecisionTree':
                predictor =load_prediction_models('DT5_loan.pkl')
                prediction = predictor.predict(prep_data)
                st.write(prediction)
            if model_choice == 'KNN':
                predictor =load_prediction_models('KNN5_Loan.pkl')
                prediction = predictor.predict(prep_data)
                st.write(prediction)
            if model_choice == 'Naive Bayes':
                predictor =load_prediction_models('NB5_Loan.pkl')
                prediction = predictor.predict(prep_data)
                st.write(prediction)
            if model_choice == 'Logistic Regresyon':
                predictor =load_prediction_models('LR5_loan.pkl')
                prediction = predictor.predict(prep_data)



                final_result = get_key(prediction,churn_l)
                st.success(final_result)
            if  prediction == 1:
                    st.success('Churn Analyis:YES.')
            else:
                    st.warning('Churn Analysis:NO.')




    if choices == 'About':
        st.subheader('About')



main()
