from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import joblib
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt
import os
import xgboost
from Libraries.lorem.code.lorem import LOREM
from Libraries.lorem.code import datamanager
import Libraries.lorem.code.util
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
from sklearn.ensemble import RandomForestClassifier
import subprocess
import sys
import dill
import shap
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from itertools import cycle, islice
import seaborn as sns
from sklearn.metrics import confusion_matrix

'''
    This class contains all the methods that are used to make predictions and to generate the explanation of prediction.
'''
class Worker:
    cat_features = ['age_group', 'race', 'income_poverty']
    
    categorical_names = {
        0: list(['No', 'Yes']),
        1: list(['No', 'Yes']),
        5: ['18 - 34 Years',
              '35 - 44 Years',
              '45 - 54 Years',
              '55 - 64 Years',
              '65+ Years'],
        6: ['Black','Hispanic','Other or Multiple','White'],
        7: ['Below Poverty', '<= $75,000, Above Poverty', '> $75,000'],
    }
    
    cat_features_h1n1 = ['age_group', 'race', 'employment_status', 'hhs_geo_region']
    
    categorical_names_h1n1 = {
        1: list(['No', 'Yes']),
        2: list(['No', 'Yes']),
        6: ['18 - 34 Years',
          '35 - 44 Years',
          '45 - 54 Years',
          '55 - 64 Years',
          '65+ Years'],
        7: ['Black','Hispanic','Other or Multiple','White'],
        8: ['Below Poverty',
            '<= $75000 Above Poverty',
            '> $75000'
          ],
        9: ['atmpeygn',
            'bhuqouqj',
            'dqpwygqj',
            'fpwskwrf',
            'kbazzjca',
            'lrircsnp',
            'lzgpxyit',
            'mlyzmhmf',
            'oxchjgsf',
            'qufhixun']
    }
    
    
    def __init__(self, seasonal, h1n1, lore=False):
        class_names = ['no_vaccine', 'vaccine']
        if not lore:
            self.my_features_seasonal = seasonal.columns.tolist()
            self.my_features_h1n1 = h1n1.columns.tolist()
            
            self.lime_explainer_seasonal = LimeTabularExplainer(
                                seasonal.values[:,:], 
                                feature_names=self.my_features_seasonal, 
                                categorical_features = self.categorical_names.keys(),
                                categorical_names = self.categorical_names, 
                                class_names = class_names, 
                                discretize_continuous=True,
                                verbose = True, 
                                mode='regression')
            self.lime_explainer_h1n1 = LimeTabularExplainer(
                                h1n1.values[:,:], 
                                feature_names = self.my_features_h1n1,
                                categorical_features = self.categorical_names_h1n1.keys(), 
                                categorical_names = self.categorical_names_h1n1,
                                class_names = class_names,
                                discretize_continuous=True,
                                verbose = True, 
                                mode='regression')
        
    '''
        This method is used to generate the Lime explanation for a prediction. In particular it explains the seasonal prediction.
    '''
    def lime_explain_seasonal(self, seasonal, index):
        self.xgb_classifier = joblib.load('./src/Models/xgb.joblib')
        class_names_seasonal = ['no_vaccine', 'vaccine']
        
        exp = self.lime_explainer_seasonal.explain_instance(seasonal.values[index], 
                                 self.xgb_classifier.predict,
                                 num_features = len(self.my_features_seasonal))
        figure = exp.as_pyplot_figure()
        plt.title("Lime Explanation")

    '''
        This method is used to generate the Lime explanation for a prediction. In particular it explains the H1N1 prediction.
    '''  
    def lime_explain_h1n1(self, h1n1, index):
        self.rf_classifier = joblib.load('./src/Models/rf.joblib')
        class_names_seasonal = ['no_vaccine', 'vaccine']
        
        exp = self.lime_explainer_h1n1.explain_instance(h1n1.values[index], 
                                 self.rf_classifier.predict,
                                 num_features = len(self.my_features_h1n1))
        figure = exp.as_pyplot_figure()
        plt.title("Lime explanation")
        
    '''
        This method is used to predict wether the respondent received the seasonal vaccine or not.
    '''
    def xgb_classifier_seasonal(self, instances, index):
        self.xgb_classifier = joblib.load('./Models/xgb.joblib')
        predictions = self.xgb_classifier.predict(instances.values[index].reshape((1,-1)))
        return predictions
    
    '''
        This method is used to predict wether the respondent received the H1N1 vaccine or not.
    '''
    def random_forest_classifier_h1n1(self, instances, index):
        self.rf_classifier = joblib.load('./Models/rf.joblib')
        predictions = self.rf_classifier.predict(instances.values[index].reshape((1,-1)))
        return predictions
    
    '''
        This method is used to extract all the predictions for a given dataset.
        It returs both the seasonal predictions and the H1N1 predictions.
    '''
    def makeAllPredictions(self, seasonal, h1n1):
        self.xgb_classifier = joblib.load('./src/Models/xgb.joblib')
        self.rf_classifier = joblib.load('./src/Models/rf.joblib')
        predictions_seasonal = self.xgb_classifier.predict(seasonal.values)
        predictions_h1n1 = self.rf_classifier.predict(h1n1.values)
        return predictions_seasonal.tolist(), predictions_h1n1.tolist()

 
    '''
        This method is used to extract the explanation made by lore. In particular it works on the seasonal dataset.
    '''
    def lore_explain_seasonal(self, seasonal, index):
        dataset_len = seasonal.shape[0]
        rdf = pd.read_csv('./Data/df_seas_XG_no_Cod.csv')

        rdf.drop('seasonal_vaccine', axis=1, inplace = True)
        rdf.append(seasonal, ignore_index=True)
        df2, feature_names2, numeric_columns2, rdf2, real_feature_names2, features_map2 = datamanager.my_prepare_dataset(rdf)
        to_explain = df2.tail(dataset_len)
        with open('./Explainers/lore_explainer_seasonal', 'rb') as f:     
            explainer = dill.load(f)
            exp = explainer.explain_instance(to_explain.values[index], 
                                            samples=100,
                                            use_weights=True)
            return exp
    
    '''
        This method is used to extract the explanation made by Shap. In particular it works on the seasonal dataset.
    '''
    def shap_explain_seasonal(seasonal,no_Cod):
        with open('./Explainers/shap_explainer_seasonal', 'rb') as f:     
            explainer = dill.load(f)
        shap_values = explainer.shap_values(seasonal)
        return explainer, shap_values
    
    '''
        This method is used to extract the explanation made by Shap. In particular it works on the H1N1 dataset.
    '''
    def shap_explain_h1n1(seasonal,no_Cod):
        with open('./Explainers/shap_explainer_h1n1', 'rb') as f:     
            explainer = dill.load(f)
        shap_values = explainer.shap_values(seasonal)
        return explainer, shap_values

    '''
        This method is used to extract the explanation made by lore. In particular it works on the H1N1 dataset.
    '''
    def lore_explain_h1n1(self, h1n1, index):
        dataset_len = h1n1.shape[0]
        rdf = pd.read_csv('./Data/df_h1n1_RF_no_Cod.csv')
        rdf.drop('h1n1_vaccine', axis=1, inplace = True)

        rdf.append(h1n1, ignore_index=True)
        df2, feature_names2, numeric_columns2, rdf2, real_feature_names2, features_map2 = datamanager.my_prepare_dataset(rdf)
        to_explain = df2.tail(dataset_len)
        
        with open('./Explainers/lore_explainer_h1n1', 'rb') as f:     
            explainer = dill.load(f)
            exp = explainer.explain_instance(to_explain.values[index], 
                                            samples=100,
                                            use_weights=True)
            return exp

    '''
        This method is used to plot a histogram that shows the distribution of the output features.
    '''
    def plotComparisonBar(df, myFeatures):
        my_colors = list( islice(cycle(['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:cyan']), None, len(df)))
        if(len(myFeatures) == 2):
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            for idx, col in enumerate(myFeatures):
                df[col].value_counts().plot(kind='bar', title=col, ax=ax[idx], color=my_colors)

            plt.xticks(rotation=0)
            fig.tight_layout()
        else:
            print("Error: Too Many features")
          
    '''
        This method is used to plot a confusion matrix with the predictions made on the seasonal dataset.
    '''
    def confMatSeas(labels_seasonal, predictions_seasonal):
        print("Seasonal predictions confusion matrix :")
        conf_matrix = confusion_matrix(labels_seasonal, predictions_seasonal)
        sns.heatmap(conf_matrix)
    
    '''
        This method is used to plot a confusion matrix with the predictions made on the H1N1 dataset.
    '''
    def confMatH1N1(labels_h1n1, predictions_h1n1):
        print("H1N1 predictions confusion matrix :")
        conf_matrix = confusion_matrix(labels_h1n1, predictions_h1n1)
        sns.heatmap(conf_matrix)

    
    def under_sampler(X_train,y_train, seed, perc):
        rus = RandomUnderSampler(random_state=seed, sampling_strategy=perc)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        return X_res, y_res
    
    
    def xgb_predict(self, X):
        return self.xgb_classifier.predict(X)
    def xgb_predict_proba(self, X):
        return self.xgb_classifier.predict_proba(X)

    def rf_predict(self, X):
        return self.rf_classifier_lore.predict(X)
    def rf_predict_proba(self, X):
        return self.rf_classifier_lore.predict_proba(X)
    
    def install(self):
        subprocess.check_call([sys.executable, "-m", "pip", "install", 'dill'])