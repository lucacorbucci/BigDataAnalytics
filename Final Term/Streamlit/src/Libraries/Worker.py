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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import streamlit as st


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
        
    
    def lime_explain_seasonal(self, seasonal, index):
        self.xgb_classifier = joblib.load('./src/Models/xgb_final.joblib')
        class_names_seasonal = ['no_vaccine', 'vaccine']
        
        exp = self.lime_explainer_seasonal.explain_instance(seasonal.values[index], 
                                 self.xgb_classifier.predict,
                                 num_features = len(self.my_features_seasonal))
        figure = exp.as_pyplot_figure()
        plt.title("Lime Explanation")
        return figure

        
    def lime_explain_h1n1(self, h1n1, index):
        self.rf_classifier = joblib.load('./src/Models/rf_classifier_final.joblib')
        class_names_seasonal = ['no_vaccine', 'vaccine']
        
        exp = self.lime_explainer_h1n1.explain_instance(h1n1.values[index], 
                                 self.rf_classifier.predict,
                                 num_features = len(self.my_features_h1n1))
        figure = exp.as_pyplot_figure()
        plt.title("Lime explanation")
        return figure
        
    
    def xgb_classifier_seasonal(self, instances, index):
        self.xgb_classifier = joblib.load('./src/Models/xgb_final.joblib')
        predictions = self.xgb_classifier.predict(instances.values[index].reshape((1,-1)))
        return predictions
    
    
    def random_forest_classifier_h1n1(self, instances, index):
        self.rf_classifier = joblib.load('./src/Models/rf_classifier_final.joblib')
        predictions = self.rf_classifier.predict(instances.values[index].reshape((1,-1)))
        return predictions
    
    def makeAllPredictions(self, seasonal, h1n1):
        self.xgb_classifier = joblib.load('./src/Models/xgb_final.joblib')
        self.rf_classifier = joblib.load('./src/Models/rf_classifier_final.joblib')
        predictions_seasonal = self.xgb_classifier.predict(seasonal.values)
        predictions_h1n1 = self.rf_classifier.predict(h1n1.values)
        return predictions_seasonal.tolist(), predictions_h1n1.tolist()

    def xgb_predict(self, X):
        return self.xgb_classifier.predict(X)
    def xgb_predict_proba(self, X):
        return self.xgb_classifier.predict_proba(X)

    def rf_predict(self, X):
        return self.rf_classifier_lore.predict(X)
    def rf_predict_proba(self, X):
        return self.rf_classifier_lore.predict_proba(X)
    
    def lore_explain_seasonal(self, seasonal, index):
        self.xgb_classifier = joblib.load('./src/Models/xgb_classifier_lore.joblib')
        dataset_len = seasonal.shape[0]
        rdf = pd.read_csv('./src/Data/df_seas_XG_no_Cod.csv')
        rdf = rdf.loc[:, ~rdf.columns.str.contains('^Unnamed')]
        rdf.drop('seasonal_vaccine', axis=1, inplace = True)
        rdf = rdf.append(seasonal, ignore_index=True)
        df2, feature_names2, numeric_columns2, rdf2, real_feature_names2, features_map2 = datamanager.my_prepare_dataset(rdf)
        to_explain = df2.tail(dataset_len)
        to_explain = to_explain.loc[:, ~df2.columns.str.contains('^Unnamed')]
       

        with open('./src/Explainers/lore_explainer_seasonal', 'rb') as f:     
            explainer = dill.load(f)
            exp = explainer.explain_instance(to_explain.values[index], 
                                            samples=100,
                                            use_weights=True)
            return exp

    def lore_explain_h1n1(self, h1n1, index):
        self.rf_classifier_lore = joblib.load('./src/Models/rf_classifier_h1n1_lore.joblib')
        dataset_len = h1n1.shape[0]
        rdf = pd.read_csv('./src/Data/df_h1n1_RF_no_Cod.csv')
        rdf.drop('h1n1_vaccine', axis=1, inplace = True)
        rdf = rdf.loc[:, ~rdf.columns.str.contains('^Unnamed')]
        rdf = rdf.append(h1n1, ignore_index=True)
        df2, feature_names2, numeric_columns2, rdf2, real_feature_names2, features_map2 = datamanager.my_prepare_dataset(rdf)
        to_explain = df2.tail(dataset_len)
        to_explain = to_explain.loc[:, ~df2.columns.str.contains('^Unnamed')]
        
        with open('./src/Explainers/lore_explainer_h1n1', 'rb') as f:     
            explainer = dill.load(f)
            exp = explainer.explain_instance(to_explain.values[index], 
                                            samples=100,
                                            use_weights=True)
            return exp

       
        
    def shap_explain_seasonal(seasonal, seasonal_sample_no_Cod, index):
        with open('./src/Explainers/shap_explainer_seasonal', 'rb') as f:     
            explainer = dill.load(f)
        shap_values = explainer.shap_values(seasonal)
        return shap.force_plot(explainer.expected_value, shap_values[index,:], seasonal_sample_no_Cod.iloc[index,:], link='logit'), shap.force_plot(explainer.expected_value, shap_values[0:], seasonal_sample_no_Cod[0:],link="logit")
    
    def shap_explain_h1n1(h1n1, h1n1_sample_no_Cod, index):
        with open('./src/Explainers/shap_explainer_h1n1', 'rb') as f:     
            explainer = dill.load(f)
        shap_values = explainer.shap_values(h1n1)
        return shap.force_plot(explainer.expected_value[1], shap_values[1][index,:], h1n1_sample_no_Cod.iloc[index,:], link='logit'), shap.force_plot(explainer.expected_value[1], shap_values[1][0:], h1n1_sample_no_Cod[0:],link="logit")

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

        
    def save_expl_shap_seasonal(self):
        self.xgb_classifier = joblib.load('./src/Models/xgb.joblib')
        seasonal_XG = pd.read_csv("./src/Data/df_seas_XG.csv")
        target_seasonal = seasonal_XG.pop('seasonal_vaccine')
        X_train, X_test, y_train, y_test = train_test_split(seasonal_XG, target_seasonal, test_size=0.3, random_state=42)
        X_test = X_test.reset_index()
        old_index = X_test.pop('index')
        xgb = XGBClassifier(colsample_bytree = 1.0,
                    gamma = 5, 
                    learning_rate = 0.1,
                    max_depth = 3,
                    min_child_weight = 10,
                    n_estimators = 180,
                    subsample = 0.8,
                   random_state=42)
        xgb.fit(X_train, y_train)
        #y_pred = xgb.predict(sample)
        shap_explainer = shap.TreeExplainer(self.xgb_classifier)  
        with open('./src/Explainers/shap_explainer_seasonal', 'wb') as f:
             foo = dill.dump(shap_explainer, f)
        return shap_explainer
    
    def under_sampler(X_train,y_train, seed, perc):
        rus = RandomUnderSampler(random_state=seed, sampling_strategy=perc)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        return X_res, y_res
    
    def save_expl_shap_h1n1(self):
        self.rf_classifier = joblib.load('./src/Models/rf.joblib')
        h1n1_RF = pd.read_csv("./src/Data/df_h1n1_RF.csv")
        target_h1n1 = h1n1_RF.pop('h1n1_vaccine')
        X_train, X_test, y_train, y_test = train_test_split(h1n1_RF, target_h1n1, test_size=0.3, random_state=42)
        X_train, y_train = Worker.under_sampler(X_train, y_train,42,0.6)
        X_test = X_test.reset_index()
        old_index = X_test.pop('index')
        rf_classifier = RandomForestClassifier(bootstrap = True, 
                                       max_depth = 8, 
                                       max_features = 'auto', 
                                       n_estimators = 50,
                                       random_state=42)
        rf_classifier.fit(X_train, y_train)
        explainer = shap.TreeExplainer(self.rf_classifier)
        with open('./src/Explainers/shap_explainer_h1n1', 'wb') as f:
             foo = dill.dump(explainer, f)
        return explainer

        
    def confMatSeas(labels_seasonal, predictions_seasonal):
        print("Seasonal predictions confusion matrix :")
        conf_matrix = confusion_matrix(labels_seasonal, predictions_seasonal)
        sns.heatmap(conf_matrix)
        st.pyplot()

    
    def confMatH1N1(labels_h1n1, predictions_h1n1):
        print("H1N1 predictions confusion matrix :")
        conf_matrix = confusion_matrix(labels_h1n1, predictions_h1n1)
        sns.heatmap(conf_matrix)
        st.pyplot()

    def plotComparisonBar(df, myFeatures):
        my_colors = list( islice(cycle(['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:cyan']), None, len(df)))
        if(len(myFeatures) == 2):
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            for idx, col in enumerate(myFeatures):
                df[col].value_counts().plot(kind='bar', title=col, ax=ax[idx], color=my_colors)

            plt.xticks(rotation=0)
            fig.tight_layout()
            return fig
        else:
            print("Error: Too Many features")
            
        # class_name = 'seasonal_vaccine'
        # df2, feature_names2, class_values2, numeric_columns2, rdf2, real_feature_names2, features_map2 = datamanager.prepare_dataset(rdf, class_name)
        # test_size = 0.30
        # random_state = 42
        # y = df2[class_name]
        # print(df2.shape)

        # X_train, X_test, y_train, y_test = train_test_split(df2[feature_names2], 
        #                                                     df2[class_name], 
        #                                                     test_size=test_size,
        #                                                     random_state=random_state)

        # _, K, _, _ = train_test_split(rdf[real_feature_names2], 
        #                             rdf[class_name], 
        #                             test_size=0.3, 
        #                             random_state=random_state)

        # self.xgb_classifier = XGBClassifier(colsample_bytree = 1.0,
        #             gamma = 5, 
        #             learning_rate = 0.1,
        #             max_depth = 3,
        #             min_child_weight = 10,
        #             n_estimators = 180,
        #             subsample = 0.8)
        
        # self.xgb_classifier.fit(X_train.values, y_train.values)

        # lore_explainer = LOREM(K.values, 
        #                self.xgb_predict, 
        #                feature_names2, 
        #                class_name, 
        #                class_values2, 
        #                numeric_columns2,
        #                features_map2,
        #                neigh_type='genetic',
        #                categorical_use_prob=True, 
        #                continuous_fun_estimation=False, 
        #                size=1000,
        #                ocr=0.1, 
        #                random_state=42, 
        #                ngen=10, 
        #                verbose=False)
        # self.install()
        # with open('./src/lore_explainer_seasonal', 'wb') as f:
        #     foo = dill.dump(lore_explainer, f)

        # return df2




        # class_name = 'h1n1_vaccine'
        # df2, feature_names2, class_values2, numeric_columns2, rdf2, real_feature_names2, features_map2 = datamanager.prepare_dataset(rdf, class_name)
        # test_size = 0.30
        # random_state = 42
        # y = df2[class_name]
        # print(df2.shape)

        # X_train, X_test, y_train, y_test = train_test_split(df2[feature_names2], 
        #                                                     df2[class_name], 
        #                                                     test_size=test_size,
        #                                                     random_state=random_state)

        # _, K, _, _ = train_test_split(rdf[real_feature_names2], 
        #                             rdf[class_name], 
        #                             test_size=0.3, 
        #                             random_state=random_state)

        # rf_classifier_H1N1 = rf_classifier = RandomForestClassifier(bootstrap = True, 
        #                                max_depth = 8, 
        #                                max_features = 'auto', 
        #                                n_estimators = 50)
        
        # rf_classifier_H1N1.fit(X_train.values, y_train.values)

        # self.rf_classifier_lore = rf_classifier_H1N1
        
        # lore_explainer = LOREM(K.values, 
        #                self.rf_predict, 
        #                feature_names2, 
        #                class_name, 
        #                class_values2, 
        #                numeric_columns2,
        #                features_map2,
        #                neigh_type='genetic',
        #                categorical_use_prob=True, 
        #                continuous_fun_estimation=False, 
        #                size=1000,
        #                ocr=0.1, 
        #                random_state=42, 
        #                ngen=10, 
        #                verbose=False)
        # self.install()
        # with open('./src/lore_explainer_h1n1', 'wb') as f:
        #     foo = dill.dump(lore_explainer, f)

        # return df2
        
        



# from sklearn.preprocessing import LabelEncoder
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# import joblib
# from lime.lime_tabular import LimeTabularExplainer
# import matplotlib.pyplot as plt
# import os
# import xgboost
# from Libraries.lorem.code.lorem import LOREM
# from Libraries.lorem.code import datamanager
# import Libraries.lorem.code.util
# import pickle
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
# from xgboost import plot_importance
# from xgboost import plot_tree
# from sklearn.ensemble import RandomForestClassifier
# import subprocess
# import sys
# import dill

# class Worker:
    
#     cat_features = ['age_group', 'race', 'income_poverty']
    
#     categorical_names = {
#         0: list(['No', 'Yes']),
#         1: list(['No', 'Yes']),
#         5: ['18 - 34 Years',
#               '35 - 44 Years',
#               '45 - 54 Years',
#               '55 - 64 Years',
#               '65+ Years'],
#         6: ['Black','Hispanic','Other or Multiple','White'],
#         7: ['Below Poverty', '<= $75,000, Above Poverty', '> $75,000'],
#     }
    
#     cat_features_h1n1 = ['age_group', 'race', 'employment_status', 'hhs_geo_region']
    
#     categorical_names_h1n1 = {
#         1: list(['No', 'Yes']),
#         2: list(['No', 'Yes']),
#         6: ['18 - 34 Years',
#           '35 - 44 Years',
#           '45 - 54 Years',
#           '55 - 64 Years',
#           '65+ Years'],
#         7: ['Black','Hispanic','Other or Multiple','White'],
#         8: ['Below Poverty',
#             '<= $75000 Above Poverty',
#             '> $75000'
#           ],
#         9: ['atmpeygn',
#             'bhuqouqj',
#             'dqpwygqj',
#             'fpwskwrf',
#             'kbazzjca',
#             'lrircsnp',
#             'lzgpxyit',
#             'mlyzmhmf',
#             'oxchjgsf',
#             'qufhixun']
#     }
    
    
    
    
    
#     def __init__(self, seasonal, h1n1, lore=False):
#         class_names = ['no_vaccine', 'vaccine']
#         if not lore:
#             self.my_features_seasonal = seasonal.columns.tolist()
#             self.my_features_h1n1 = h1n1.columns.tolist()
            
#             self.lime_explainer_seasonal = LimeTabularExplainer(
#                                 seasonal.values[:,:], 
#                                 feature_names=self.my_features_seasonal, 
#                                 categorical_features = self.categorical_names.keys(),
#                                 categorical_names = self.categorical_names, 
#                                 class_names = class_names, 
#                                 discretize_continuous=True,
#                                 verbose = True, 
#                                 mode='regression')
#             self.lime_explainer_h1n1 = LimeTabularExplainer(
#                                 h1n1.values[:,:], 
#                                 feature_names = self.my_features_h1n1,
#                                 categorical_features = self.categorical_names_h1n1.keys(), 
#                                 categorical_names = self.categorical_names_h1n1,
#                                 class_names = class_names,
#                                 discretize_continuous=True,
#                                 verbose = True, 
#                                 mode='regression')
        
    
#     def lime_explain_seasonal(self, seasonal, index):
#         self.xgb_classifier = joblib.load('./src/Models/xgb.joblib')
#         class_names_seasonal = ['no_vaccine', 'vaccine']
        
#         exp = self.lime_explainer_seasonal.explain_instance(seasonal.values[index], 
#                                  self.xgb_classifier.predict,
#                                  num_features = len(self.my_features_seasonal))
#         figure = exp.as_pyplot_figure()
#         plt.title("Lime Explanation")
#         return figure

        
#     def lime_explain_h1n1(self, h1n1, index):
#         self.rf_classifier = joblib.load('./src/Models/rf.joblib')
#         class_names_seasonal = ['no_vaccine', 'vaccine']
        
#         exp = self.lime_explainer_h1n1.explain_instance(h1n1.values[index], 
#                                  self.rf_classifier.predict,
#                                  num_features = len(self.my_features_h1n1))
#         figure = exp.as_pyplot_figure()
#         plt.title("Lime explanation")
#         return figure
        
    
#     def xgb_classifier_seasonal(self, instances, index):
#         self.xgb_classifier = joblib.load('./Models/xgb.joblib')
#         predictions = self.xgb_classifier.predict(instances.values[index].reshape((1,-1)))
#         return predictions
    
    
#     def random_forest_classifier_h1n1(self, instances, index):
#         self.rf_classifier = joblib.load('./Models/rf.joblib')
#         predictions = self.rf_classifier.predict(instances.values[index].reshape((1,-1)))
#         return predictions
    
#     def makeAllPredictions(self, seasonal, h1n1):
#         self.xgb_classifier = joblib.load('./src/Models/xgb.joblib')
#         self.rf_classifier = joblib.load('./src/Models/rf.joblib')
#         predictions_seasonal = self.xgb_classifier.predict(seasonal.values)
#         predictions_h1n1 = self.rf_classifier.predict(h1n1.values)
#         return predictions_seasonal.tolist(), predictions_h1n1.tolist()

#     def xgb_predict(self, X):
#         return self.xgb_classifier.predict(X)
#     def xgb_predict_proba(self, X):
#         return self.xgb_classifier.predict_proba(X)

#     def rf_predict(self, X):
#         return self.rf_classifier_lore.predict(X)
#     def rf_predict_proba(self, X):
#         return self.rf_classifier_lore.predict_proba(X)
    
#     def install(self):
#         subprocess.check_call([sys.executable, "-m", "pip", "install", 'dill'])
 
#     def lore_explain_seasonal(self, seasonal, index):
#         dataset_len = seasonal.shape[0]
#         rdf = pd.read_csv('./src/Data/df_seas_XG_no_Cod.csv')

#         rdf.drop('seasonal_vaccine', axis=1, inplace = True)
#         rdf.append(seasonal, ignore_index=True)
#         df2, feature_names2, numeric_columns2, rdf2, real_feature_names2, features_map2 = datamanager.my_prepare_dataset(rdf)
#         to_explain = df2.tail(dataset_len)
#         with open('./src/Explainers/lore_explainer_seasonal', 'rb') as f:     
#             explainer = dill.load(f)
#             exp = explainer.explain_instance(to_explain.values[index], 
#                                             samples=100,
#                                             use_weights=True)
#             return exp

#     def lore_explain_h1n1(self, h1n1, index):
#         dataset_len = h1n1.shape[0]
#         rdf = pd.read_csv('./src/Data/df_h1n1_RF_no_Cod.csv')
#         rdf.drop('h1n1_vaccine', axis=1, inplace = True)

#         rdf.append(h1n1, ignore_index=True)
#         df2, feature_names2, numeric_columns2, rdf2, real_feature_names2, features_map2 = datamanager.my_prepare_dataset(rdf)
#         to_explain = df2.tail(dataset_len)
        
#         with open('./src/Explainers/lore_explainer_h1n1', 'rb') as f:     
#             explainer = dill.load(f)
#             exp = explainer.explain_instance(to_explain.values[index], 
#                                             samples=100,
#                                             use_weights=True)
#             return exp

        
#     def Lore_explain_h1n1(self, h1n1, index):
#         self.rf_classifier_lore = joblib.load('./src/Models/rf_classifier_h1n1_lore.joblib')
        



#         # class_name = 'seasonal_vaccine'
#         # df2, feature_names2, class_values2, numeric_columns2, rdf2, real_feature_names2, features_map2 = datamanager.prepare_dataset(rdf, class_name)
#         # test_size = 0.30
#         # random_state = 42
#         # y = df2[class_name]
#         # print(df2.shape)

#         # X_train, X_test, y_train, y_test = train_test_split(df2[feature_names2], 
#         #                                                     df2[class_name], 
#         #                                                     test_size=test_size,
#         #                                                     random_state=random_state)

#         # _, K, _, _ = train_test_split(rdf[real_feature_names2], 
#         #                             rdf[class_name], 
#         #                             test_size=0.3, 
#         #                             random_state=random_state)

#         # self.xgb_classifier = XGBClassifier(colsample_bytree = 1.0,
#         #             gamma = 5, 
#         #             learning_rate = 0.1,
#         #             max_depth = 3,
#         #             min_child_weight = 10,
#         #             n_estimators = 180,
#         #             subsample = 0.8)
        
#         # self.xgb_classifier.fit(X_train.values, y_train.values)

#         # lore_explainer = LOREM(K.values, 
#         #                self.xgb_predict, 
#         #                feature_names2, 
#         #                class_name, 
#         #                class_values2, 
#         #                numeric_columns2,
#         #                features_map2,
#         #                neigh_type='genetic',
#         #                categorical_use_prob=True, 
#         #                continuous_fun_estimation=False, 
#         #                size=1000,
#         #                ocr=0.1, 
#         #                random_state=42, 
#         #                ngen=10, 
#         #                verbose=False)
#         # self.install()
#         # with open('./src/lore_explainer_seasonal', 'wb') as f:
#         #     foo = dill.dump(lore_explainer, f)

#         # return df2




#         # class_name = 'h1n1_vaccine'
#         # df2, feature_names2, class_values2, numeric_columns2, rdf2, real_feature_names2, features_map2 = datamanager.prepare_dataset(rdf, class_name)
#         # test_size = 0.30
#         # random_state = 42
#         # y = df2[class_name]
#         # print(df2.shape)

#         # X_train, X_test, y_train, y_test = train_test_split(df2[feature_names2], 
#         #                                                     df2[class_name], 
#         #                                                     test_size=test_size,
#         #                                                     random_state=random_state)

#         # _, K, _, _ = train_test_split(rdf[real_feature_names2], 
#         #                             rdf[class_name], 
#         #                             test_size=0.3, 
#         #                             random_state=random_state)

#         # rf_classifier_H1N1 = rf_classifier = RandomForestClassifier(bootstrap = True, 
#         #                                max_depth = 8, 
#         #                                max_features = 'auto', 
#         #                                n_estimators = 50)
        
#         # rf_classifier_H1N1.fit(X_train.values, y_train.values)

#         # self.rf_classifier_lore = rf_classifier_H1N1
        
#         # lore_explainer = LOREM(K.values, 
#         #                self.rf_predict, 
#         #                feature_names2, 
#         #                class_name, 
#         #                class_values2, 
#         #                numeric_columns2,
#         #                features_map2,
#         #                neigh_type='genetic',
#         #                categorical_use_prob=True, 
#         #                continuous_fun_estimation=False, 
#         #                size=1000,
#         #                ocr=0.1, 
#         #                random_state=42, 
#         #                ngen=10, 
#         #                verbose=False)
#         # self.install()
#         # with open('./src/lore_explainer_h1n1', 'wb') as f:
#         #     foo = dill.dump(lore_explainer, f)

#         # return df2