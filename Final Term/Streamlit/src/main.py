import numpy as np
import pandas as pd
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from Libraries.PreProcessing import PreProcessing
from Libraries.Worker import Worker
import shap
import streamlit.components.v1 as components
from sklearn.metrics import accuracy_score 

# This method generates the title that is shown at the top of the page.
def showTitle():
    st.markdown("<h1 style='text-align: center; color: black;'>Flu Shot Learning: Predict H1N1 and Seasonal Flu Vaccines</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: black;'>University of Pisa - Big Data Analytics Final Term - Academic Year 2020/21</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'>MaLuCS Team</h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: black;'></h3>", unsafe_allow_html=True)
    st.markdown('## Upload your data')
    st.markdown("In this phase data will be cleaned and we will create two datasets, one for the prediction of the **seasonal** vaccine and one for the prediction of the **H1N1** vaccine.")
    st.markdown("The two dataset will be shown after a few seconds.")

# This method is used to visualize the two datasets
def showDatasets(seasonal, h1n1):
    st.markdown('### Seasonal Dataset')
    st.markdown(f"The seasonal dataset has {seasonal.shape[0]} rows and {seasonal.shape[1]} columns")
    st.write(seasonal)
    st.markdown('### H1N1 Dataset')
    st.markdown(f"The H1N1 dataset has {h1n1.shape[0]} rows and {h1n1.shape[1]} columns")
    st.write(h1n1)

# This method is used to clean the uploaded dataset and to return the cleaned datasets
def preProcess(uploaded_file):
    df = pd.read_csv(uploaded_file, sep=",")
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    seasonal, h1n1, labels_seasonal, labels_h1n1 = PreProcessing.TestSetCleaning(df)
    st.write(df)
    return df, seasonal, h1n1, labels_seasonal, labels_h1n1

# This method is used to make the predictions both for the seasonal and the H1N1 dataset.
# It also shows the accuracy for the predictions.
def makePredictions(seasonal, h1n1, labels_seasonal, labels_h1n1):
    worker = Worker(seasonal, h1n1)
    predictions_seasonal, predictions_h1n1 = worker.makeAllPredictions(seasonal, h1n1)
    p_seasonal = predictions_seasonal
    p_h1n1 = predictions_h1n1
    accuracy_seasonal = accuracy_score(predictions_seasonal, labels_seasonal)
    accuracy_h1n1 = accuracy_score(predictions_h1n1, labels_h1n1)
    predictions_seasonal = ['Vaccinated' if prediction == 1 else 'Not Vaccinated' for prediction in predictions_seasonal]
    predictions_h1n1 = ['Vaccinated' if prediction == 1 else 'Not Vaccinated' for prediction in predictions_h1n1]
    labels_seasonal = ['Vaccinated' if prediction == 1 else 'Not Vaccinated' for prediction in labels_seasonal]
    labels_h1n1 = ['Vaccinated' if prediction == 1 else 'Not Vaccinated' for prediction in labels_h1n1]
    df_predictions = pd.DataFrame({'Seasonal Predictions': predictions_seasonal,
                                    'Seasonal True Labels': labels_seasonal, 
                                    'H1N1 Predictions': predictions_h1n1,
                                    'H1N1 True Label': labels_h1n1})
    st.markdown('## Predictions on the entire dataset')
    st.markdown('To make these predictions we used a XGBoost Classifier on the seasonal dataset and a Random Forest Classifier on the H1N1 dataset')
    st.table(df_predictions)
    st.write(f"The accuracy for the seasonal prediction is {accuracy_seasonal}")
    st.write(f"The accuracy for the H1N1 prediction is {accuracy_h1n1}")
    return df_predictions

# This method is used to visualize the plot produced by shap in a scrollable div
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><div style='overflow-x:scroll;overflow-y:hidden;width:2500px;height:150px'>{plot.html()}</div>"
    components.html(shap_html, scrolling=True)

# This method is used to visualize the plot produced by shap in a scrollable div
def st_shap_other_plot(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><div style='overflow-x:scroll;overflow-y:hidden;width:2500px;height:450px'>{plot.html()}</div>"
    components.html(shap_html, height=500, scrolling=True)

# This method is used to generate the explanation of an instance using Shap
def shap_explanation(seasonal, h1n1, selected_index, full_df):
    st.markdown("## Explanation with Shap")
    seasonal_noCod, h1n1_noCod, labels_seasonal, labels_h1n1 = PreProcessing.TestSetCleaningNoCod(full_df)
    worker = Worker(seasonal, h1n1)

    st.markdown("### Seasonal Prediction Explanation with Shap")
    g1, g2 = Worker.shap_explain_seasonal(seasonal, seasonal_noCod, selected_index)
    st_shap(g1)
    st_shap_other_plot(g2)

    st.markdown("### H1N1 Prediction Explanation with Shap")
    g1, g2 = Worker.shap_explain_h1n1(h1n1, h1n1_noCod, selected_index)
    st_shap(g1)
    st_shap_other_plot(g2)

# This method is used to generate the explanation of an instance using Lime
def lime_explanation(seasonal, h1n1, selected_index):
    worker = Worker(seasonal, h1n1)
    st.markdown("## Explanation with Lime")
    
    st.markdown('### Seasonal Prediction')
    figure = worker.lime_explain_seasonal(seasonal, selected_index)
    st.pyplot(figure)
    st.markdown('### H1N1 Prediction')
    figure_h1n1 = worker.lime_explain_h1n1(h1n1, selected_index)
    st.pyplot(figure_h1n1)

# This method is used to generate the explanation of an instance using Lore
def lore_explanation(seasonal, h1n1, selected_index):
    seasonal_noCod, h1n1_noCod, labels_seasonal, labels_h1n1 = PreProcessing.TestSetCleaningNoCod(full_df)
    worker = Worker(seasonal_noCod, h1n1_noCod, True)
    st.markdown("## Explanation with Lore")

    st.markdown('### Seasonal Prediction Explanation with Lore')
    explanation = worker.lore_explain_seasonal(seasonal_noCod, selected_index)
    st.write(explanation)
    st.markdown('### H1N1 Prediction Explanation with Lore')
    explanation = worker.lore_explain_h1n1(h1n1_noCod, selected_index)
    st.write(explanation)

# This method is used to produce the explanation and to visualize them in the page
def askExplanation(df_predictions, seasonal, h1n1, full_df):
    st.write('## Choose the instance you want to explain', df_predictions)
    selected_index = st.selectbox('Select a row:', df_predictions.index)
    selected_row_seasonal = seasonal.loc[selected_index]
    selected_row_h1n1 = h1n1.loc[selected_index]

    st.write(f'You selected {selected_index}')
    st.write("Seasoal instance to explain:")
    st.write(selected_row_seasonal)
    st.write("H1N1 instance to explain:")
    st.write(selected_row_h1n1)

    if st.button('Explain Prediction'):
        lime_explanation(seasonal, h1n1, selected_index)
        lore_explanation(seasonal, h1n1, selected_index)
        shap_explanation(seasonal, h1n1, selected_index, full_df)

'''
    * *********************************************************************************** *
'''
showTitle()
uploaded_file = st.file_uploader("Choose a file", type=['txt', 'csv'])

if uploaded_file is not None:
    full_df, seasonal, h1n1, labels_seasonal, labels_h1n1 = preProcess(uploaded_file)
    showDatasets(seasonal, h1n1)
    st.pyplot(Worker.plotComparisonBar(full_df, ['h1n1_vaccine', 'seasonal_vaccine']))
    df_predictions = makePredictions(seasonal, h1n1, labels_seasonal, labels_h1n1)
    askExplanation(df_predictions, seasonal, h1n1, full_df)




