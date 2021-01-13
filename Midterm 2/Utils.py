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

class Utils:

    '''
        Using this function you can print all values which are present in each column provided in input

        Params:
        - cols: list of the columns of the dataset you want to print
        - df: the DataFrame which contains the dataset

    '''

    def split_train_test(df, target, seed, size=0.3):
        atrb = [col for col in df.columns if col != target]
        X = df[atrb]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
        return X, y, X_train, X_test, y_train, y_test,atrb

    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Mean validation score: {0:.5f} (std: {1:.5f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def DT_grid_sh(split_list, param_list):
        clf_tmp = DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf_tmp = clf_tmp.fit(split_list[2], split_list[4])
        grid_search = GridSearchCV(clf_tmp, param_grid=param_list)
        grid_search.fit(split_list[0], split_list[1])
        Utils.report(grid_search.cv_results_, n_top=3)

    def Model_Eval_DT(X_train,y_train , X_test, y_test, min_split, min_leaf):
        clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=min_split, min_samples_leaf=min_leaf)
        clf = clf.fit(X_train,y_train)
        acc_train = []
        F1_train = []
        print ("Training set")
        y_pred_train = clf.predict(X_train)
        print('Accuracy %s' % accuracy_score(y_train, y_pred_train))
        acc_train.append(accuracy_score(y_train, y_pred_train))
        print('F1-score %s' % f1_score(y_train, y_pred_train, average=None))
        F1_train.append(f1_score(y_train, y_pred_train, average=None))
        print(classification_report(y_train, y_pred_train))
        dict_train=classification_report(y_train, y_pred_train, output_dict=True)
        acc_test = []
        F1_test = []
        print ("Test set")
        y_pred_test = clf.predict(X_test)
        print('Accuracy %s' % accuracy_score(y_test, y_pred_test))
        acc_test.append(accuracy_score(y_test, y_pred_test))
        print('F1-score %s' % f1_score(y_test, y_pred_test, average=None))
        F1_test.append(f1_score(y_test, y_pred_test, average=None))
        print(classification_report(y_test, y_pred_test))
        dict_test = classification_report(y_test, y_pred_test, output_dict=True)
        return clf, acc_train, F1_train, acc_test, F1_test, dict_train['0']['precision'], dict_train['1']['precision'] , dict_train['0']['recall'], dict_train['1']['recall'], dict_test['0']['precision'], dict_test['1']['precision'] , dict_test['0']['recall'], dict_test['1']['recall']

    def Model_Eval_DT_weight(X_train, y_train, X_test, y_test, min_split, min_leaf, weight):
        clf = DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=min_split,
                                     min_samples_leaf=min_leaf, class_weight = weight)
        clf = clf.fit(X_train, y_train)
        acc_train = []
        F1_train = []
        print ("Training set")
        y_pred_train = clf.predict(X_train)
        print('Accuracy %s' % accuracy_score(y_train, y_pred_train))
        acc_train.append(accuracy_score(y_train, y_pred_train))
        print('F1-score %s' % f1_score(y_train, y_pred_train, average=None))
        F1_train.append(f1_score(y_train, y_pred_train, average=None))
        print(classification_report(y_train, y_pred_train))
        dict_train = classification_report(y_train, y_pred_train, output_dict=True)
        acc_test = []
        F1_test = []
        print ("Test set")
        y_pred_test = clf.predict(X_test)
        print('Accuracy %s' % accuracy_score(y_test, y_pred_test))
        acc_test.append(accuracy_score(y_test, y_pred_test))
        print('F1-score %s' % f1_score(y_test, y_pred_test, average=None))
        F1_test.append(f1_score(y_test, y_pred_test, average=None))
        print(classification_report(y_test, y_pred_test))
        dict_test = classification_report(y_test, y_pred_test, output_dict=True)
        return clf, acc_train, F1_train, acc_test, F1_test, dict_train['0']['precision'], dict_train['1']['precision'], \
               dict_train['0']['recall'], dict_train['1']['recall'], dict_test['0']['precision'], dict_test['1'][
                   'precision'], dict_test['0']['recall'], dict_test['1']['recall']


    def print_single_ROC_curve(clf, split_list,title, label, color):
        y_score = clf.predict_proba(split_list[3])
        fpr, tpr, _ = roc_curve(split_list[5], y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=3, label=label % (roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title(title, fontsize=16)
        plt.legend(loc="lower right", fontsize=14, frameon=False)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.show()

    def print_multy_ROC_curve(clf_g, split_list_g, title, label_list, color_list, n):
        for i in range(0,n):
            y_score = clf_g[i].predict_proba(split_list_g[i][3])
            fpr, tpr, _ = roc_curve(split_list_g[i][5], y_score[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color_list[i], lw=3, label=label_list[i] % (roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.title(title, fontsize=16)
        plt.legend(loc="lower right", fontsize=14, frameon=False)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.show()

    def plotSingleFeatureBarPlot(df, feature_name, rot=0):
        fig, ax = plt.subplots()
        my_colors = list(islice(cycle(['tab:blue', 'tab:orange', 'tab:red', 'tab:green', 'tab:purple', 'tab:cyan']), None, len(df)))
        df[feature_name].value_counts().sort_index().plot(
                kind='bar', title=feature_name, sort_columns=False, color=my_colors)
        plt.xticks(rotation=rot)
        fig.tight_layout()

    def show_PCA(X_train, y_train):
        X_train1 = StandardScaler().fit_transform(X_train)
        pca = PCA(n_components=2)
        pca.fit(X_train1)
        X_pca = pca.transform(X_train1)
        print("Shape :")
        print(X_pca.shape)
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=plt.cm.prism, edgecolor='k', alpha=0.7)
        plt.title("PCA", fontweight='bold')
        plt.show()

    def under_sampler(X_train,y_train, seed, perc):
        rus = RandomUnderSampler(random_state=seed, sampling_strategy=perc)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        return X_res, y_res

    def adasyn_sampler(X_train, y_train, seed, perc):
        ad = ADASYN(random_state=seed, sampling_strategy=perc)
        X_res, y_res = ad.fit_resample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        return X_res, y_res

    def under_over_sampler(X_train,y_train, seed, perc_1, perc_2):
        rus = RandomUnderSampler(random_state=seed, sampling_strategy=perc_1)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        ros = RandomOverSampler(random_state=seed, sampling_strategy=perc_2)
        X_res, y_res = ros.fit_resample(X_res, y_res)
        print('Resampled dataset shape %s' % Counter(y_res))
        return X_res, y_res

    def under_smote_sampler(X_train,y_train, seed, perc_1, perc_2):
        rus = RandomUnderSampler(random_state=seed, sampling_strategy=perc_1)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        sm = SMOTE(random_state=seed, sampling_strategy=perc_2)
        X_res, y_res = sm.fit_resample(X_res, y_res)
        print('Resampled dataset shape %s' % Counter(y_res))
        return X_res, y_res

    def over_smote_sampler(X_train, y_train, seed, perc_1, perc_2):
        ros = RandomOverSampler(random_state=seed, sampling_strategy=perc_1)
        X_res, y_res = ros.fit_resample(X_train, y_train)
        sm = SMOTE(random_state=seed, sampling_strategy=perc_2)
        X_res, y_res = sm.fit_resample(X_res, y_res)
        print('Resampled dataset shape %s' % Counter(y_res))
        return X_res, y_res

    def over_under_sampler(X_train, y_train, seed, perc_1, perc_2):
        rus = RandomOverSampler(random_state=seed, sampling_strategy=perc_1)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        ros = RandomUnderSampler(random_state=seed, sampling_strategy=perc_2)
        X_res, y_res = ros.fit_resample(X_res, y_res)
        print('Resampled dataset shape %s' % Counter(y_res))
        return X_res, y_res


    def over_sampler(X_train, y_train, seed, perc):
        ros =  RandomOverSampler(random_state=seed, sampling_strategy=perc)
        X_res, y_res = ros.fit_resample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        return X_res, y_res

    def smote_sampler(X_train, y_train, seed, perc):
        sm = SMOTE(random_state=seed, sampling_strategy=perc)
        X_res, y_res = sm.fit_resample(X_train, y_train)
        print('Resampled dataset shape %s' % Counter(y_res))
        return X_res, y_res

    def acc_F1_table(acc,F1,color_list,x_list,n):
        plt.figure(figsize=(18, 9))
        plt.title('Accuracy and F1-Score', size=32, fontweight='bold', color='black')
        plt.plot(range(1, n+1), acc, color=color_list[0], linestyle='solid', lw=3, marker='.', markerfacecolor=color_list[0],
                 markersize=18, label='Accuracy')
        fc0 = []
        fc1 = []
        for i in range(0, n):
            fc0.append(F1[i][0])
            fc1.append(F1[i][1])
        plt.plot(range(1, n+1), fc0, color=color_list[1], linestyle='solid', lw=3, marker='.', markerfacecolor=color_list[1],
                 markersize=18, label='Class 0: F1-score')
        plt.plot(range(1, n+1), fc1, color=color_list[2], linestyle='solid', lw=3, marker='.',
                 markerfacecolor=color_list[2], markersize=18, label='Class 1: F1-score')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xticks(list(range(1,n+1)), x_list,
                   size=15, fontweight='bold', rotation=90)
        plt.legend(loc='best', fontsize=20)
        plt.ylabel('', size=19)

    def pre_rec_table(pre_0,pre_1, rec_0, rec_1, color_list,x_list,n):
        plt.figure(figsize=(18, 9))
        plt.title('Precision and Recall', size=32, fontweight='bold', color='black')
        plt.plot(range(1, n+1), pre_0, color=color_list[0], linestyle='solid', lw=3, marker='.', markerfacecolor=color_list[0],
                 markersize=18, label='Precision 0')
        plt.plot(range(1, n + 1), pre_1, color=color_list[1], linestyle='solid', lw=3, marker='.',
                 markerfacecolor=color_list[1],
                 markersize=18, label='Precision 1')
        plt.plot(range(1, n + 1), rec_0, color=color_list[2], linestyle='solid', lw=3, marker='.',
                 markerfacecolor=color_list[2],
                 markersize=18, label='Recall 0')
        plt.plot(range(1, n + 1), rec_1, color=color_list[3], linestyle='solid', lw=3, marker='.',
                 markerfacecolor=color_list[3],
                 markersize=18, label='Recall 1')
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.xticks(list(range(1,n+1)), x_list,
                   size=23, fontweight='bold', rotation=90)
        plt.legend(loc='best', fontsize=20)
        plt.ylabel('', size=19)

    def val_diff(acc1,acc2,F1_1,F1_2):
        r = acc1 - acc2
        print('Accuracy difference %f' % r)
        l = F1_1[0] - F1_2[0]
        r = F1_1[1] - F1_2[1]
        print('F1-score difference [ %f' % l + ' %f' % r + ']')

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

    def visual_RFECV_DT(X_train, y_train, min_split, min_leaf):
        visualizer = RFECV(DecisionTreeClassifier(criterion='gini', max_depth=None, min_samples_split=min_split, min_samples_leaf=min_leaf))
        visualizer.fit(X_train, y_train)
        visualizer.show()
        return visualizer

    def visual_RFECV(clf, X_train, y_train):
        visualizer = RFECV(clf)
        visualizer.fit(X_train, y_train)
        visualizer.show()
        return visualizer

    def RFECV_to_DF(visualizer, df, target):
        r = visualizer.ranking_
        features = [col for col in df.columns if col != target]
        feature_rank = []
        for i in range(len(features)):
            feature_rank = feature_rank + [[features[i], r[i]]]
        column_val = ['feature', 'rank']
        return pd.DataFrame(data=feature_rank, columns=column_val)

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

    def best_over(SL, min_split, min_leaf, seed):
        acc = []
        F1 = []
        precision_0 = []
        precision_1 = []
        recall_0 = []
        recall_1 = []
        for i in range (3,11):
            perc=i/10
            sample = RandomOverSampler(random_state=seed, sampling_strategy=perc)
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

    def best_smote(SL, min_split, min_leaf, seed):
            acc = []
            F1 = []
            precision_0 = []
            precision_1 = []
            recall_0 = []
            recall_1 = []
            for i in range(3, 11):
                perc = i / 10
                sample = SMOTE(random_state=seed, sampling_strategy=perc)
                X_sample, y_sample = sample.fit_resample(SL[2], SL[4])
                print ()
                print('Resampled dataset shape %s' % Counter(y_sample))
                print('Sampling perc :  %s' % perc)
                print ()
                tmp = Utils.Model_Eval_DT_forBalancing(X_sample, y_sample, SL[3], SL[5], min_split, min_leaf)
                acc = acc + tmp[0]
                F1 = F1 + tmp[1]
                precision_0 = precision_0 + [tmp[2]]
                precision_1 = precision_1 + [tmp[3]]
                recall_0 = recall_0 + [tmp[4]]
                recall_1 = recall_1 + [tmp[5]]

            Utils.acc_F1_table(acc, F1, ['red', 'blue', 'green'],
                               ['30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], 8)
            Utils.pre_rec_table(precision_0, precision_1, recall_0, recall_1,
                                ['red', 'blue', 'green', 'darkorange'],
                                ['30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'], 8)

    def best_adasyn(SL, min_split, min_leaf, seed):
        acc = []
        F1 = []
        precision_0 = []
        precision_1 = []
        recall_0 = []
        recall_1 = []
        for i in range(4, 11):
            perc = i / 10
            sample = ADASYN(random_state=seed, sampling_strategy=perc)
            X_sample, y_sample = sample.fit_resample(SL[2], SL[4])
            print ()
            print('Resampled dataset shape %s' % Counter(y_sample))
            print('Sampling perc :  %s' % perc)
            print ()
            tmp = Utils.Model_Eval_DT_forBalancing(X_sample, y_sample, SL[3], SL[5], min_split, min_leaf)
            acc = acc + tmp[0]
            F1 = F1 + tmp[1]
            precision_0 = precision_0 + [tmp[2]]
            precision_1 = precision_1 + [tmp[3]]
            recall_0 = recall_0 + [tmp[4]]
            recall_1 = recall_1 + [tmp[5]]

        Utils.acc_F1_table(acc, F1, ['red', 'blue', 'green'],
                           ['40%', '50%', '60%', '70%', '80%', '90%', '100%'], 7)
        Utils.pre_rec_table(precision_0, precision_1, recall_0, recall_1,
                            ['red', 'blue', 'green', 'darkorange'],
                            ['40%', '50%', '60%', '70%', '80%', '90%', '100%'], 7)

    def best_under_over(SL, min_split, min_leaf, seed):
            acc = []
            F1 = []
            precision_0 = []
            precision_1 = []
            recall_0 = []
            recall_1 = []
            tags = []
            for i in range(3, 10):
                perc_1 = i / 10
                sample = RandomUnderSampler(random_state=seed, sampling_strategy=perc_1)
                X_sample, y_sample = sample.fit_resample(SL[2], SL[4])
                for j in range (i+1, 11):
                    perc_2 = j / 10
                    sample = RandomOverSampler(random_state=seed, sampling_strategy=perc_2)
                    X_sample, y_sample = sample.fit_resample(X_sample, y_sample)
                    print ()
                    print('Resampled dataset shape %s' % Counter(y_sample))
                    print('perc_1 :  %s' % perc_1)
                    print('perc_2 :  %s' % perc_2)
                    print ()
                    tmp = Utils.Model_Eval_DT_forBalancing(X_sample, y_sample, SL[3], SL[5], min_split, min_leaf)
                    acc = acc + tmp[0]
                    F1 = F1 + tmp[1]
                    precision_0 = precision_0 + [tmp[2]]
                    precision_1 = precision_1 + [tmp[3]]
                    recall_0 = recall_0 + [tmp[4]]
                    recall_1 = recall_1 + [tmp[5]]

            Utils.acc_F1_table(acc, F1, ['red', 'blue', 'green'],
                               ['0.3% 0.4%', '0.3% 0.5%', '0.3% 0.6%', '0.3% 0.7%', '0.3% 0.8%', '0.3% 0.9%', '0.3% 1.0%', '0.4% 0.5%', '0.4% 0.6%', '0.4% 0.7%', '0.4% 0.8%', '0.4% 0.9%', '0.4% 1.0%', '0.5% 0.6%', '0.5% 0.7%', '0.5% 0.8%', '0.5% 0.9%', '0.5% 1.0%', '0.6% 0.7%', '0.6% 0.8%', '0.6% 0.9%', '0.6% 1.0%', '0.7% 0.8%', '0.7% 0.9%', '0.7% 1.0%', '0.8% 0.9%', '0.8% 1.0%', '0.9% 1.0%'], 28)

            Utils.pre_rec_table(precision_0, precision_1, recall_0, recall_1,['red', 'blue', 'green', 'darkorange'], ['0.3% 0.4%', '0.3% 0.5%', '0.3% 0.6%', '0.3% 0.7%', '0.3% 0.8%', '0.3% 0.9%', '0.3% 1.0%', '0.4% 0.5%', '0.4% 0.6%', '0.4% 0.7%', '0.4% 0.8%', '0.4% 0.9%', '0.4% 1.0%', '0.5% 0.6%', '0.5% 0.7%', '0.5% 0.8%', '0.5% 0.9%', '0.5% 1.0%', '0.6% 0.7%', '0.6% 0.8%', '0.6% 0.9%', '0.6% 1.0%', '0.7% 0.8%', '0.7% 0.9%', '0.7% 1.0%', '0.8% 0.9%', '0.8% 1.0%', '0.9% 1.0%'], 28)

    def best_over_under(SL, min_split, min_leaf, seed):
            acc = []
            F1 = []
            precision_0 = []
            precision_1 = []
            recall_0 = []
            recall_1 = []
            tags = []
            for i in range(3, 10):
                perc_1 = i / 10
                sample = RandomOverSampler(random_state=seed, sampling_strategy=perc_1)
                X_sample, y_sample = sample.fit_resample(SL[2], SL[4])
                for j in range (i+1, 11):
                    perc_2 = j / 10
                    sample = RandomUnderSampler(random_state=seed, sampling_strategy=perc_2)
                    X_sample, y_sample = sample.fit_resample(X_sample, y_sample)
                    print ()
                    print('Resampled dataset shape %s' % Counter(y_sample))
                    print('perc_1 :  %s' % perc_1)
                    print('perc_2 :  %s' % perc_2)
                    print ()
                    tmp = Utils.Model_Eval_DT_forBalancing(X_sample, y_sample, SL[3], SL[5], min_split, min_leaf)
                    acc = acc + tmp[0]
                    F1 = F1 + tmp[1]
                    precision_0 = precision_0 + [tmp[2]]
                    precision_1 = precision_1 + [tmp[3]]
                    recall_0 = recall_0 + [tmp[4]]
                    recall_1 = recall_1 + [tmp[5]]

            Utils.acc_F1_table(acc, F1, ['red', 'blue', 'green'],
                               ['0.3% 0.4%', '0.3% 0.5%', '0.3% 0.6%', '0.3% 0.7%', '0.3% 0.8%', '0.3% 0.9%', '0.3% 1.0%', '0.4% 0.5%', '0.4% 0.6%', '0.4% 0.7%', '0.4% 0.8%', '0.4% 0.9%', '0.4% 1.0%', '0.5% 0.6%', '0.5% 0.7%', '0.5% 0.8%', '0.5% 0.9%', '0.5% 1.0%', '0.6% 0.7%', '0.6% 0.8%', '0.6% 0.9%', '0.6% 1.0%', '0.7% 0.8%', '0.7% 0.9%', '0.7% 1.0%', '0.8% 0.9%', '0.8% 1.0%', '0.9% 1.0%'], 28)

            Utils.pre_rec_table(precision_0, precision_1, recall_0, recall_1,['red', 'blue', 'green', 'darkorange'], ['0.3% 0.4%', '0.3% 0.5%', '0.3% 0.6%', '0.3% 0.7%', '0.3% 0.8%', '0.3% 0.9%', '0.3% 1.0%', '0.4% 0.5%', '0.4% 0.6%', '0.4% 0.7%', '0.4% 0.8%', '0.4% 0.9%', '0.4% 1.0%', '0.5% 0.6%', '0.5% 0.7%', '0.5% 0.8%', '0.5% 0.9%', '0.5% 1.0%', '0.6% 0.7%', '0.6% 0.8%', '0.6% 0.9%', '0.6% 1.0%', '0.7% 0.8%', '0.7% 0.9%', '0.7% 1.0%', '0.8% 0.9%', '0.8% 1.0%', '0.9% 1.0%'], 28)

    def best_under_smote(SL, min_split, min_leaf, seed):
            acc = []
            F1 = []
            precision_0 = []
            precision_1 = []
            recall_0 = []
            recall_1 = []
            tags = []
            for i in range(3, 10):
                perc_1 = i / 10
                sample = RandomUnderSampler(random_state=seed, sampling_strategy=perc_1)
                X_sample, y_sample = sample.fit_resample(SL[2], SL[4])
                for j in range (i+1, 11):
                    perc_2 = j / 10
                    sample = SMOTE(random_state=seed, sampling_strategy=perc_2)
                    X_sample, y_sample = sample.fit_resample(X_sample, y_sample)
                    print ()
                    print('Resampled dataset shape %s' % Counter(y_sample))
                    print('perc_1 :  %s' % perc_1)
                    print('perc_2 :  %s' % perc_2)
                    print ()
                    tmp = Utils.Model_Eval_DT_forBalancing(X_sample, y_sample, SL[3], SL[5], min_split, min_leaf)
                    acc = acc + tmp[0]
                    F1 = F1 + tmp[1]
                    precision_0 = precision_0 + [tmp[2]]
                    precision_1 = precision_1 + [tmp[3]]
                    recall_0 = recall_0 + [tmp[4]]
                    recall_1 = recall_1 + [tmp[5]]

            Utils.acc_F1_table(acc, F1, ['red', 'blue', 'green'],
                               ['0.3% 0.4%', '0.3% 0.5%', '0.3% 0.6%', '0.3% 0.7%', '0.3% 0.8%', '0.3% 0.9%', '0.3% 1.0%', '0.4% 0.5%', '0.4% 0.6%', '0.4% 0.7%', '0.4% 0.8%', '0.4% 0.9%', '0.4% 1.0%', '0.5% 0.6%', '0.5% 0.7%', '0.5% 0.8%', '0.5% 0.9%', '0.5% 1.0%', '0.6% 0.7%', '0.6% 0.8%', '0.6% 0.9%', '0.6% 1.0%', '0.7% 0.8%', '0.7% 0.9%', '0.7% 1.0%', '0.8% 0.9%', '0.8% 1.0%', '0.9% 1.0%'], 28)

            Utils.pre_rec_table(precision_0, precision_1, recall_0, recall_1,['red', 'blue', 'green', 'darkorange'], ['0.3% 0.4%', '0.3% 0.5%', '0.3% 0.6%', '0.3% 0.7%', '0.3% 0.8%', '0.3% 0.9%', '0.3% 1.0%', '0.4% 0.5%', '0.4% 0.6%', '0.4% 0.7%', '0.4% 0.8%', '0.4% 0.9%', '0.4% 1.0%', '0.5% 0.6%', '0.5% 0.7%', '0.5% 0.8%', '0.5% 0.9%', '0.5% 1.0%', '0.6% 0.7%', '0.6% 0.8%', '0.6% 0.9%', '0.6% 1.0%', '0.7% 0.8%', '0.7% 0.9%', '0.7% 1.0%', '0.8% 0.9%', '0.8% 1.0%', '0.9% 1.0%'], 28)

    def best_over_smote(SL, min_split, min_leaf, seed):
            acc = []
            F1 = []
            precision_0 = []
            precision_1 = []
            recall_0 = []
            recall_1 = []
            tags = []
            for i in range(3, 10):
                perc_1 = i / 10
                sample = RandomOverSampler(random_state=seed, sampling_strategy=perc_1)
                X_sample, y_sample = sample.fit_resample(SL[2], SL[4])
                for j in range (i+1, 11):
                    perc_2 = j / 10
                    sample = SMOTE(random_state=seed, sampling_strategy=perc_2)
                    X_sample, y_sample = sample.fit_resample(X_sample, y_sample)
                    print ()
                    print('Resampled dataset shape %s' % Counter(y_sample))
                    print('perc_1 :  %s' % perc_1)
                    print('perc_2 :  %s' % perc_2)
                    print ()
                    tmp = Utils.Model_Eval_DT_forBalancing(X_sample, y_sample, SL[3], SL[5], min_split, min_leaf)
                    acc = acc + tmp[0]
                    F1 = F1 + tmp[1]
                    precision_0 = precision_0 + [tmp[2]]
                    precision_1 = precision_1 + [tmp[3]]
                    recall_0 = recall_0 + [tmp[4]]
                    recall_1 = recall_1 + [tmp[5]]

            Utils.acc_F1_table(acc, F1, ['red', 'blue', 'green'],
                               ['0.3% 0.4%', '0.3% 0.5%', '0.3% 0.6%', '0.3% 0.7%', '0.3% 0.8%', '0.3% 0.9%', '0.3% 1.0%', '0.4% 0.5%', '0.4% 0.6%', '0.4% 0.7%', '0.4% 0.8%', '0.4% 0.9%', '0.4% 1.0%', '0.5% 0.6%', '0.5% 0.7%', '0.5% 0.8%', '0.5% 0.9%', '0.5% 1.0%', '0.6% 0.7%', '0.6% 0.8%', '0.6% 0.9%', '0.6% 1.0%', '0.7% 0.8%', '0.7% 0.9%', '0.7% 1.0%', '0.8% 0.9%', '0.8% 1.0%', '0.9% 1.0%'], 28)

            Utils.pre_rec_table(precision_0, precision_1, recall_0, recall_1,['red', 'blue', 'green', 'darkorange'], ['0.3% 0.4%', '0.3% 0.5%', '0.3% 0.6%', '0.3% 0.7%', '0.3% 0.8%', '0.3% 0.9%', '0.3% 1.0%', '0.4% 0.5%', '0.4% 0.6%', '0.4% 0.7%', '0.4% 0.8%', '0.4% 0.9%', '0.4% 1.0%', '0.5% 0.6%', '0.5% 0.7%', '0.5% 0.8%', '0.5% 0.9%', '0.5% 1.0%', '0.6% 0.7%', '0.6% 0.8%', '0.6% 0.9%', '0.6% 1.0%', '0.7% 0.8%', '0.7% 0.9%', '0.7% 1.0%', '0.8% 0.9%', '0.8% 1.0%', '0.9% 1.0%'], 28)

