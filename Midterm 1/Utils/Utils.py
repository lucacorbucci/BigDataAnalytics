import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import LabelEncoder
import numpy as np
from itertools import cycle, islice

class Utils:

    '''
        Using this function you can print all values which are present in each column provided in input

        Params:
        - cols: list of the columns of the dataset you want to print
        - df: the DataFrame which contains the dataset

    '''

    def printValues(cols, df):
        for col in cols:
            print(df[col].value_counts())
            print("")

    '''
        Using this function you can replace the missing values which are present in each column provided in input with the mode

        Params:
        - cols: list of the columns of the dataset you want to print
        - df: the DataFrame which contains the dataset

    '''
    def dfMode(cols, df):
        for col in cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

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

    '''
        Single stacked bar plot.
        
        Params:
        - col: DataFrame's feature.
        - target: target feature.
        - df: the DataFrame which contains the dataset
        - ax(optional) : matplotlib axes object to attach plot to
    '''

    def vaccination_rate_plot(col, target, df, ax=None):
        counts = (df[[target, col]]
                  .groupby([target, col])
                  .size()
                  .unstack(target)
                  )
        group_counts = counts.sum(axis='columns')
        props = counts.div(group_counts, axis='index')

        props.plot(kind="barh", stacked=True, ax=ax,)
        ax.invert_yaxis()
        ax.legend().remove()

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
        Using this function you can plot a stacked bar chart like this one: https://matplotlib.org/_images/sphx_glr_bar_stacked_001.png
        (the bars are rotated).
        
        Params:
        - feature1: the first target feature
        - feature2: the second target feature
        - myfeatures: the columns of the dataset we want to consider
        - df: the DataFrame which contains the dataset
    '''
    def plotConditionalHistogram_SingleImage(feature1, feature2, myFeatures, df):
        # 0 = No; 1 = Yes.
        columns = 2 if feature2 != '' else 1

        fig, ax = plt.subplots(
            len(myFeatures), columns, figsize=(9, len(myFeatures)*3)
        )
        for idx, col in enumerate(myFeatures):
            if (len(myFeatures) == 1):
                Utils.vaccination_rate_plot(col, feature1, df, ax=ax[0])
                if (feature2 != ""):
                    Utils.vaccination_rate_plot(col, feature2, df, ax=ax[1])
            else:
                Utils.vaccination_rate_plot(

                    col, feature1, df, ax=ax[idx, 0] if columns == 2 else ax[idx])
                if (feature2 != ""):
                    Utils.vaccination_rate_plot(
                        col, feature2, df, ax=ax[idx, 1])

        if (len(myFeatures) == 1):
            ax[0].legend(
                loc='lower center', bbox_to_anchor=(0.5, 1.05), title=feature1
            )
            if (feature2 != ""):
                ax[1].legend(
                    loc='lower center', bbox_to_anchor=(0.5, 1.05), title=feature2
                )
        else:

            ax[0, 0].legend(
                loc='lower center', bbox_to_anchor=(0.5, 1.05), title=feature1
            ) if columns == 2 else ax[0].legend(
                loc='lower center', bbox_to_anchor=(0.5, 1.05), title=feature1
            )
            if (feature2 != ""):
                ax[0, 1].legend(
                    loc='lower center', bbox_to_anchor=(0.5, 1.05), title=feature2
                )
        
        fig.tight_layout()

    '''
        Using this function you can plot a plot like this: https://matplotlib.org/_images/sphx_glr_bar_stacked_001.png 
        (the plot is normalized)

        Params:
        - df: the DataFrame which contains the dataset
        - feature_name: the column of the dataset you want to consider in your plot
        - target: the target you want to consider
        - title: the title that is shown in the plot
        - save: True if you want to save the plot, default value is False
    '''
    def plotSingleCrossTab(df, feature_name, target, title, save=False):
        plt.figure(figsize=(7, 5))
        crosstab = pd.crosstab(df[feature_name], df[target])
        crosstab = crosstab.div(crosstab.sum(1).astype(float), axis=0)

        b1 = plt.bar(crosstab.index, crosstab[0], width=0.5)
        b2 = plt.bar(crosstab.index, crosstab[1],
                     bottom=crosstab[0], width=0.5)
        plt.xlabel(feature_name, fontsize=12)
        plt.xticks(rotation=30, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend((b1[0], b2[0]), ('No', 'Yes'), title=title)
        plt.show()
        if save:
            plt.savefig('plots/' + feature_name + '_' +
                        'target' + '.png', bbox_inches="tight")

    '''
        Using this function you can plot the correlation Heatmap.
        
        Params:
        - df: the DataFrame which contains the dataset
        - method: the method we want to use to compute the correlation, default is "pearson"        
    '''
    def plotHeatMapCorrelation(df, annot=False, method='pearson'):
        corr = df.corr(method=method)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        g = sns.heatmap(corr, center=0, mask=mask, xticklabels=True, yticklabels=True,
                        square=True, linewidths=1, cbar_kws={"shrink": .5}, annot=annot, fmt='.2f', cmap='YlGnBu')
        sns.despine()
        g.figure.set_size_inches(15, 15)

        plt.show()

    '''
        This function compute the correlation between a feature and the target feature.
        
        Params:
        - df: the DataFrame which contains the dataset
        - features: the features you want to consider
        - targetFeature: the target feature you want to consider in the computation of the correlation
    '''
    def printCorrelation(df, features, targetFeature):
        for x in features:
            print('\nCorrelation by:', x)
            corr = df[[x, targetFeature]].groupby(x).mean()
            print(corr)

    '''
        This function can be used to encode a categorical and ordinal features.
        
        Example:
        Feature age_group: 
            Utils.encodeOrdinal(dataset, 'age_group', {"18 - 34 Years": 0, "35 - 44 Years": 1, "45 - 54 Years": 2, "55 - 64 Years": 3, "65+ Years": 4})
        
        Params:
        - df: the DataFrame which contains the dataset
        - feature: the feature we are considering
        - mapping: a mapping that consider the order of the values of this feature
    '''
    def encodeOrdinal(df, feature, mapping):
        df[feature] = df[feature].apply(lambda x: mapping[x])

    '''
        This function can be used to encode the categorical features that are not ordinal.
        
        Example:
            Utils.encodeValues(dataset, 'race')
        
        Params:    
        - df: the DataFrame which contains the dataset
        - feature: the feature we want to encode
    '''
    def encodeValues(df, feature):
        labelencoder = LabelEncoder()
        df[feature] = labelencoder.fit_transform(df[feature])

    '''
        This function prints the most relevant correlation between the features in a list.
    
        Params:
        - df: the DataFrame which contains the dataset
        - cols_list: feature list
    '''
    def printPearsonCoefficentForTupleList(df, cols_list):
        list_tuple = list(combinations(cols_list, 2))
        for el in list_tuple:
            p_coef = df[el[0]].corr(df[el[1]], method="pearson")
            abs_p_coef = abs(p_coef)
            if (abs_p_coef < 0.2):
                continue
            else:
                string_result = "weak correlation"
                if (abs_p_coef > 0.3):
                    string_result = "moderate correlation"
                if (abs_p_coef > 0.4):
                    string_result = "strong correlation"
                if (abs_p_coef > 0.7):
                    string_result = "very strong correlation"
                print(el[0] + " - " + el[1] + " -> Pearson coeff:  " +
                      str(p_coef) + " " + string_result)
                print()

    '''
        This function prints the features with a correlation greater than the threshold separated from the others.
        
        -list_to_print: list of couples [abs(correlation), feature]
        -threshold: [0.0,1.0]
    '''

    def my_print_list(list_to_print, threshold):
        always_over_threshold = True
        correlated = []
        for el in list_to_print:
            if (el[0] > threshold):
                correlated.append(el[1])
            if (el[0] < threshold and always_over_threshold):
                always_over_threshold = False
                print("----------------")
                print ()
            print(el[1] + " : " + str(el[0]))
            print ()
        return correlated

    '''
        This function prints the ranked correlation of all features of the dataset with respect to one of them.
        
        Params:
        - df: the DataFrame which contains the dataset
        -label: a feature
        -threshold(optional): [0.0,1.0]     
    '''

    def print_ranked_correlation(df, label, threshold=0.10):
        columns = list(df)
        my_list = []
        for x in columns:
            res = (df[x]
                   .corr(df[label], method="pearson")
                   )
            abs_res = abs(res)
            if (x.find("vaccine") == -1 and x.find("respondent") == -1):
                my_list.append([abs_res, x])
        my_list.sort(reverse=True)
        return Utils.my_print_list(my_list, threshold)

    '''
        This function prints the correlation likelihood between a target variable and the features in a list.
         
        Params:
        - target: target feature
        - cols: feature list
        - df: the DataFrame which contains the dataset
    '''

    def printLu(target, cols, df):
        tab = df[cols]
        for x in tab:
            if x != target:
                print('\n correlation by '+target+' :', x)
                print(tab[[x, target]].groupby(x).mean())
