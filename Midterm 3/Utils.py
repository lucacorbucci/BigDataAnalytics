import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import numpy as np
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from itertools import cycle, islice
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import CondensedNearestNeighbour
from collections import Counter
from collections import defaultdict
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from yellowbrick.model_selection import RFECV
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from imblearn.over_sampling import ADASYN
from numpy import where
import matplotlib.colors as mcolors
import shap
import random
from scipy import spatial

class Utils:

    '''
        This function splits the dataset into X_train, X_test, y_train, y_test.

        Params:
        - df: the DataFrame which contains the dataset
        -target: target attribute of the splitting
        -seed: random seed to have the same splitting
        -size: size of the test part of the splitting

    '''

    def split_train_test(df, target, seed, size=0.3):
        atrb = [col for col in df.columns if col != target]
        X = df[atrb]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
        return X, y, X_train, X_test, y_train, y_test,atrb

    '''
        Grid sh. best results report

        Params:
        - results: CV results
        -n_top: number of top results 
        
    '''

    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Mean validation score: {0:.5f} (std: {1:.5f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    '''
        Grid sh. for Decision Trees

        Params:
        - split_list: list containing the splitting of the original dataset
        - param_list: list of parameters to test

    '''

    def DT_grid_sh(split_list, param_list):
        clf_tmp = DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf_tmp = clf_tmp.fit(split_list[2], split_list[4])
        grid_search = GridSearchCV(clf_tmp, param_grid=param_list)
        grid_search.fit(split_list[0], split_list[1])
        Utils.report(grid_search.cv_results_, n_top=3)


    '''
           Using this function you can plot a plot like this one: https://pandas.pydata.org/pandas-docs/stable/_images/pandas-DataFrame-plot-bar-1.png

           Params:
           - df: the DataFrame which contains the dataset
           - feature_name: the column of the dataset you want to plot
           - rot(optional): x label rotation
    '''

    def plotSingleFeatureBarPlot(df, feature_name, rot=0):
        fig, ax = plt.subplots()
        my_colors = list(islice(cycle(['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:cyan']), None, len(df)))
        df[feature_name].value_counts().sort_index().plot(
                kind='bar', title=feature_name, sort_columns=False, color=my_colors)
        plt.xticks(rotation=rot)
        fig.tight_layout()

    """
        This function apply a % random under sampler to our splitted data giving to us the result
        
        Params:
        -X_train
        -y_train
        -seed: random seed to have the same sampling
        -perc: sampling_strategy
        
    """

    def under_sampler(X_train,y_train, seed, perc):
        rus = RandomUnderSampler(random_state=seed, sampling_strategy=perc)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        return X_res, y_res


    def feature_importance_DT(clf,atrb):
        x, y = (list(x) for x in zip(*sorted(zip(clf.feature_importances_, atrb),reverse=True)))
        plt.rcdefaults()
        fig, ax = plt.subplots()
        ax.barh(y, x, align='center')
        ax.set_yticks(y)
        ax.set_yticklabels(y)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        plt.show()

    """
        Apply and visualize RFECV on our dataset returning the trained model ( only for Decision Tree)

        Params:
        - X_train
        - y_train
        - min_split: DecisionTreeClassifier parameter
        - min_leaf: DecisionTreeClassifier parameter

    """

    def visual_RFECV_DT(X_train, y_train, min_split, min_leaf):
        visualizer = RFECV(DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=min_split, min_samples_leaf=min_leaf))
        visualizer.fit(X_train, y_train)
        visualizer.show()
        return visualizer

    """
        Apply and visualize RFECV on our dataset returning the trained model

        Params:
        - clf: not trained classifier
        - X_train
        - y_train
        
    """

    def visual_RFECV(clf, X_train, y_train):
        visualizer = RFECV(clf)
        visualizer.fit(X_train, y_train)
        visualizer.show()
        return visualizer

    """
        Use the result of RFECV on our dataset to create a ranking DataFrame for the features

        Params:
        - visualizer: RFECV result
        - df: original DataFrame
        - target: target feature

    """

    def RFECV_to_DF(visualizer, df, target):
        r = visualizer.ranking_
        features = [col for col in df.columns if col != target]
        feature_rank = []
        for i in range(len(features)):
            feature_rank = feature_rank + [[features[i], r[i]]]
        column_val = ['feature', 'rank']
        return pd.DataFrame(data=feature_rank, columns=column_val)

    """
        Plots the results of RFECV on our dataset

        Params:
        - visualizer: RFECV result
        - df: original DataFrame
        - target: target feature

    """

    def print_features_rank(visualizer, df, target):
        rank_df = Utils.RFECV_to_DF(visualizer,df,target)
        rank_df = rank_df.sort_values(by=['rank'], ascending=False)
        selected = rank_df[rank_df['rank'] == 1]
        rejected = rank_df[rank_df['rank'] != 1]
        fig, ax = plt.subplots()
        ax.set_ylim(rank_df['rank'].max()+1, 0)
        ax.grid(False)
        ax.plot(rank_df["feature"], rank_df["rank"], color='darkblue',zorder=-1, linewidth=3.0)
        ax.scatter(selected["feature"], selected["rank"], c='g', s=50)
        ax.scatter(rejected["feature"], rejected["rank"], c='r', s=50)
        plt.xticks(rotation=90)
        legend_elements = [Line2D([0], [0], marker='o', color='darkred', label='Discarded',
                          markerfacecolor='red', markersize=10),Line2D([0], [0], marker='o', color='darkgreen', label='Selected',
                          markerfacecolor='green', markersize=10)]
        ax.legend(handles=legend_elements, loc='upper left')
        plt.show()

    """
    Study our unblanced dataset using under sampling at different % (Decision Tree only)
    
    Params:  
    -SL: splitting of our dataset in train and test
    - min_split: DecisionTreeClassifier parameter
    - min_leaf: DecisionTreeClassifier parameter
    - seed: random seed to have the same sampling
    
    """

    def best_under(SL, min_split, min_leaf, seed):
        acc = []
        F1 = []
        precision_0 = []
        precision_1 = []
        recall_0 = []
        recall_1 = []
        for i in range (3,11):
            perc=i/10
            sample = RandomUnderSampler(random_state=seed, sampling_strategy=perc)
            X_sample, y_sample = sample.fit_resample(SL[2], SL[4])
            print ()
            print('Resampled dataset shape %s' % Counter(y_sample))
            print('Sampling perc :  %s' % perc )
            print ()
            tmp = Utils.Model_Eval_DT_forBalancing(X_sample,y_sample,SL[3],SL[5], min_split,min_leaf)
            acc = acc + tmp[0]
            F1= F1 + tmp[1]
            precision_0 = precision_0 + [tmp[2]]
            precision_1 = precision_1 + [tmp[3]]
            recall_0 = recall_0 + [tmp[4]]
            recall_1 = recall_1 + [tmp[5]]

        Utils.acc_F1_table(acc, F1, ['red', 'blue', 'green'],['30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], 8)
        Utils.pre_rec_table(precision_0, precision_1, recall_0, recall_1,
                            ['red', 'blue', 'green', 'darkorange'],
                            ['30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], 8)

    """
     Model evaluation suitable for blancing (Decision Tree only)
    
    Params:    
    - X_train
    - y_train
    - X_test
    - y_test
    - min_split: DecisionTreeClassifier parameter
    - min_leaf: DecisionTreeClassifier parameter


    """


    def Model_Eval_DT_forBalancing(X_train,y_train , X_test, y_test,  min_split, min_leaf):
        clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=min_split, min_samples_leaf=min_leaf)
        clf = clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        acc_test = []
        F1_test = []
        print('Accuracy %s' % accuracy_score(y_test, y_pred))
        acc_test.append(accuracy_score(y_test, y_pred))
        print('F1-score %s' % f1_score(y_test, y_pred, average=None))
        F1_test.append(f1_score(y_test, y_pred, average=None))
        print(classification_report(y_test, y_pred))
        dict_test = classification_report(y_test, y_pred, output_dict=True)
        return acc_test, F1_test, dict_test['0']['precision'], dict_test['1']['precision'] , dict_test['0']['recall'], dict_test['1']['recall']


    """
    Print the ROC curve of a classifier
    
    Params:
    - y_test
    - y_pred
    - classifier_name: string containing the name of the classifier

    """

    def computeRoc(y_test, y_pred, classifier_name):
        fpr, tpr,_= roc_curve(y_test, y_pred)
        roc_auc= auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=3, label='$AUC-Before$ = %.3f' % (roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title(f"ROC curve {classifier_name}", fontsize=16)
        plt.legend(loc="lower right", fontsize=14, frameon=False)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.show()

    """
    Print the ROC curve of different classifiers
    
    Params:
    - y_test
    - rocList: dict of the ROC of different classifiers

    """

    def computeFullRoc(y_test, rocList):
        colors = []
        for k in mcolors.TABLEAU_COLORS.keys():
            colors.append(k)

        for index, key in enumerate(rocList):
            fpr, tpr,_= roc_curve(y_test, rocList[key])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr,
                     tpr,
                     color=colors[index],
                     lw=3, 
                     label=f'${key}$ = %.3f' % (roc_auc))


        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=10)
        plt.ylabel('True Positive Rate', fontsize=10)
        plt.title(f"ROC curve Comparison", fontsize=10)
        plt.legend(loc="upper left", fontsize=12, bbox_to_anchor=(1.05, 1))
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.show()


    """
    Print the ROC curve of 2 instances of the same type of classifier
        
    Params:
    - y_pred
    - y_test
    - y_pred_baseline
    - classifier_name: string containing the name of the classifier

    """
    
    def plotRoc(y_pred, y_test, y_pred_baseline, classifier_name):
        fpr, tpr,_= roc_curve(y_test, y_pred)
        fpr_baseline, tpr_baseline, _= roc_curve(y_test, y_pred_baseline)

        roc_auc = auc(fpr, tpr)
        roc_auc_baseline = auc(fpr_baseline, tpr_baseline)

        plt.plot(fpr,tpr,color='darkorange',lw=3, label='$Best \ Parameters$ = %.3f' % (roc_auc))
        plt.plot(fpr_baseline, tpr_baseline, color='green', lw=3, label='$Baseline$ = %.3f' % (roc_auc_baseline))

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title(f"ROC curve {classifier_name}", fontsize=16)
        plt.legend(loc="lower right", fontsize=14, frameon=False)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.show()

    def multy_dependence_plot(target, shap_values, X_test, interaction_list):
        for i in interaction_list:
            shap.dependence_plot(target, shap_values, X_test, interaction_index=i)

    """
    Print the classification_report

    Params:
    - y_pred
    - y_test

    """

    def printReport(y_test, y_pred):
        print(classification_report(y_test, y_pred))

    def split_test_res (y_test, y_pred):
        list_00 = []
        list_11 = []
        list_10 = []
        list_01 = []
        for i in range(len(y_pred)):
            if y_pred[i] == y_test.iloc[i] and y_pred[i] == 0: list_00 = list_00 + [i]
            if y_pred[i] == y_test.iloc[i] and y_pred[i] == 1: list_11 = list_11 + [i]
            if y_pred[i] != y_test.iloc[i] and y_pred[i] == 1: list_10 = list_10 + [i]
            if y_pred[i] != y_test.iloc[i] and y_pred[i] == 0: list_01 = list_01 + [i]
        return list_00, list_11, list_10, list_01

    
    '''
        Generate the set of instances to be explained using an explainer

        Params:
        - y_pred: list with all the predictions made by our model
        - y_test_list: list with the true values from the test set
        - pred: Predicted value we want to consider. Example 1 or 0
        - true: True value we want to consider. Example 1 or 0
        - num_instances: Number of instances we want to return. 
        Default is 4.
        
        Return:
        - A list with the set of instances to be explained
    '''
    def generate_instances_to_explain(y_pred, y_test_list, pred, true, num_instances = 4):
        random.seed(42)
        instances_to_be_explained = []
        while len(instances_to_be_explained) < num_instances:
            index = random.randint(0, len(y_pred))
            if(y_pred[index] == pred and y_test_list[index] == true):
                instances_to_be_explained.append(index)
        return instances_to_be_explained

    
    def vector2dict(x, feature_names):
        return {k: v for k, v in zip(feature_names, x)}


    def record2str(x, feature_names, numeric_columns):
        xd = Utils.vector2dict(x, feature_names)
        print(xd)
        s = '{ '
        for att, val in xd.items():

            if att not in numeric_columns and val == 0.0:
                continue
            if att in numeric_columns:
                s += '%s = %s, ' % (att, val)
            else:
                att_split = att.split('=')
                s += '%s = %s, ' % (att_split[0], att_split[1])

        s = s[:-2] + ' }'
        return s

    '''
        Generate the set of instances that are similar to the one provided

        Params:
        - x_test: test set
        - y_pred: predicted target
        - y_test: true target 
        - original: original instance for which we want to find the similar ones
        - num_instances: number of instances we want to generate
        
        Return:
        - 
    '''
    def get_similar_instances(x_test, y_pred, y_test, original = 0, num_instance = 5):
        a = np.array(list(x_test.iloc[original]))
        distances = []
        arr = []
        indexes = []
        for index, row in enumerate(x_test.itertuples()):
            l = [row.doctor_recc_seasonal, row.health_worker, row.opinion_seas_vacc_effective, row.opinion_seas_risk, row.opinion_seas_sick_from_vacc, row.age_group, row.race, row.income_poverty]
            dist = 1 - spatial.distance.cosine(a, l)
            if(dist < 1 and y_pred[original] != y_pred[index] and y_pred[index] != y_test[index]):
                distances.append(dist)
                arr.append(l)
                indexes.append(index)

        instances_to_explain = []
        distances.sort(reverse = True)
        selected_indexes = []
        idx = 0
        temp_index = []
        
        while(len(selected_indexes) < num_instance and idx < len(distances)):
            max_value = distances[idx]
            idx += 1
            index = distances.index(max_value)
            if(index not in temp_index):
                instances_to_explain.append(arr[index])
                temp_index.append(index)
                selected_indexes.append(indexes[index])

        return instances_to_explain, selected_indexes