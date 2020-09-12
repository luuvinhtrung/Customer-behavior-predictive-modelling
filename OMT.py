"""
Luu Vinh Trung
OneMount Technical test
August 16th 2020
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
import re

def get_full_data():
    """
    Returns a processed dataset as dataframe from dataset.csv file input. The output including derived features of month purchase count and amount.
    """
    try:
        df = pd.read_csv("dataset.csv")
        count_lst = np.array([])#to store the number of viewed products of each transaction
        purchase_amount_lst = np.array([])
        for index,row in df.iterrows():
            purchase_list = row['transaction_info']
            purchase_count = purchase_list.count('}, {')+1 #purchase count of the record
            purchase_amount = get_total_amount(purchase_list) #purchase amount of the record
            count_lst = np.append(count_lst,purchase_count)#derived column data
            purchase_amount_lst = np.append(purchase_amount_lst,purchase_amount)#derived column data
        #create new columns for the dataset to train model
        df['purchase_count'] = count_lst.tolist()
        df['purchase_amount'] = purchase_amount_lst.tolist()
        df['month_year'] = pd.to_datetime(df['date']).dt.to_period('M')#as there are only Feb, Mar, Apr, May, June of 2018, it is better to convert them into month only
        df = df.drop(['transaction_info','date'],axis=1)
        #df = pd.pivot_table(df, index=["csn"],columns=["month_year"],values=["purchase_amount","purchase_count"],aggfunc= np.sum)#.reset_index(level=[0])
        df_purchase_amount = pd.pivot_table(df, index=["csn"],columns=["month_year"],values=["purchase_amount"],aggfunc= np.sum)#pivot to group purchase amount by csn
        df_purchase_amount.columns = df_purchase_amount.columns.get_level_values(1)
        df_purchase_amount = df_purchase_amount.add_prefix('purchase_amount_')
        df_purchase_count = pd.pivot_table(df, index=["csn"],columns=["month_year"],values=["purchase_count"],aggfunc= np.sum)#pivot to group purchase count by csn
        df_purchase_count.columns = df_purchase_count.columns.get_level_values(1)
        df_purchase_count = df_purchase_count.add_prefix('purchase_count_')
    except Exception as exception:
        print(exception)
        return None
    return df_purchase_amount, df_purchase_count#two independent dataset to facilitate the analysis

def get_total_amount(purchase_list):
    """
    Returns total amount of money spent on a list of purchase in a record.
    """
    try:
        purchase_listed = purchase_list.split('}, {')
        total_amount = 0
        for purchase in purchase_listed:
            purchase_detail = purchase.split(',')
            str = (purchase_detail[1].split(':'))[1]
            quantity = float(str)
            str = purchase_detail[2].split(':')[1]
            str = str.strip("]}")
            price = float(str)
            total_amount+=quantity*price
    except Exception as exception:
        print(exception)
        return None
    return total_amount

def get_month_data_to_predict(df, month):
    """
        Returns a processed dataset as dataframe including data of the month to predict, and the two previous month data
    """
    try:
        df.drop(df.columns[month-1:month+1], axis=1, inplace=True)#data of the month to predict must be available for every row
        df.dropna(subset=[df.columns[0], df.columns[1]], inplace=True, how='all')#at least one of two previous month data must be available
        df = df.fillna(0)#as it is requested to predict who will make at least 1 purchase, the labels can be classified to be
        df.loc[df[df.columns[2]] > 0, df.columns[2]] = 1#0: no purchase and 1: purchase
    except Exception as exception:
        print(exception)
        return None
    return df

def remove_correlative_feature(data, corr, threshold):
    """
        Returns a processed dataset as dataframe not including the removed feature due to the feature correlation.
        The removal decision is based on a correlation threshold through scanning the correlation matrix corr.
    """
    try:
        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i + 1, corr.shape[0]):
                if corr.iloc[i, j] >= threshold:
                    if columns[j]:
                        columns[j] = False
        selected_columns = data.columns[columns]
        data = data[selected_columns]
    except Exception as exception:
        print(exception)
        return None
    return data

def get_specific_month_data_to_predict(month):
    """
        Returns a processed dataset as a combined dataframe from get_month_data_to_predict()
        for both purchase amount and count.
    """
    try:
        df_purchase_amount, df_purchase_count = get_full_data()
        df_purchase_amount = get_month_data_to_predict(df_purchase_amount,month)
        df_purchase_count = get_month_data_to_predict(df_purchase_count,month)
        purchase_data = pd.concat([df_purchase_count, df_purchase_amount], axis=1, join='inner')#full and original dataset
    except Exception as exception:
        print(exception)
        return None
    return purchase_data

def get_training_and_test_sets(purchase_data):
    """
        Returns X_train, y_train, X_test, y_test to train and valid data.
    """
    try:
        y = purchase_data.iloc[:, 5]
        purchase_data.drop(purchase_data.columns[[2, 5]], axis=1, inplace=True)
        X = purchase_data
    except Exception as exception:
        print(exception)
        return None
    return train_test_split(X, y, test_size=0.25, random_state=0)

def random_forest_classifier_predict(purchase_data):
    """
        Implements Random Forest Classifier with optimized hyperparams by GridSearchCV, shows f1 and roc_auc scores of the prediction on test set
    """
    try:
        X_train, X_test, y_train, y_test = get_training_and_test_sets(purchase_data)
        clf = RandomForestClassifier(random_state=42, max_features='auto', n_estimators=20, max_depth=4, criterion='gini')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(accuracy_score(y_test, y_pred))#0.74
        print(roc_auc_score(y_test, y_pred))#0.74

        # param_grid = {
        #     'n_estimators': [10, 20, 30, 50, 100],
        #     'max_features': ['auto', 'sqrt', 'log2'],
        #     'max_depth' : [4,6,8,10],
        #     'criterion' :['gini', 'entropy']
        # }

        # CV_clf = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)
        # CV_clf.fit(X_train, y_train)
        # print(CV_clf.best_params_)
        # print(CV_clf.best_score_)
    except Exception as exception:
        print(exception)

def xgboost_classifier_predict(purchase_data):
    """
        Implements XGBoosting Classifier with optimized hyperparams by GridSearchCV, shows r2 and roc_auc scores of the prediction on test set
    """
    try:
        X_train, X_test, y_train, y_test = get_training_and_test_sets(purchase_data)
        estimator = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                      colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                      importance_type='gain', interaction_constraints='',
                      learning_rate=0.05, max_delta_step=0, max_depth=5,
                      min_child_weight=1, monotone_constraints='()',
                      n_estimators=60, n_jobs=4, nthread=4, num_parallel_tree=1,
                      random_state=42, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                      seed=42, subsample=1, tree_method='exact', validate_parameters=1,
                      verbosity=None)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        print(r2_score(y_test, y_pred))#-0.03252337791244386 the negative score shows the disadvantage of the model
        print(roc_auc_score(y_test, y_pred))#0.7410161555741303

        # parameters = {
        #     'max_depth': range(5, 10, 1),
        #     'n_estimators': range(60, 80, 140),
        #     'learning_rate': [0.2, 0.01, 0.05]
        # }
        #
        # CV_clf = GridSearchCV(estimator=estimator, param_grid=parameters, scoring = 'roc_auc', n_jobs = 10, cv = 10, verbose=True)
        # CV_clf.fit(X_train, y_train)
        # print(CV_clf.best_estimator_)
        # print(CV_clf.best_params_)
        # print(CV_clf.best_score_)

        #y_pred = CV_clf.predict(X_test)
        #print(CV_clf.best_score_)
    except Exception as exception:
        print(exception)

def eda_purchase_count(data):
    """
        Show plot of the difference between Purchase and No Purchase count for data analysis, and the difference is little,
        that means the prediction accuracy should be quite higher than 50%
    """
    try:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.countplot(x=data.columns[5], data=data)
        plt.title("Count of Purchases")
        plt.show()
    except Exception as exception:
        print(exception)

def eda_purchase_var_correlation(corr):
    """
        Show plot of the correlation between feature of X
    """
    try:
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.title("Purchase var correlation")
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)
        plt.show()
    except Exception as exception:
        print(exception)

def eda_purchase_and_average(data):
    """
        Show plot of the correlation between median of purchase count/amount and the purchase decision in the month to predict
    """
    try:
        for col in data.columns:
            fig, ax = plt.subplots(1, figsize=(8, 6))
            sns.boxplot(x=data.columns[4], y=col, data=data)
            ax.set_ylim(0,100)#(0,300000)
            plt.title("Purchase and Avg correlation")
            plt.show()
    except Exception as exception:
        print(exception)


purchase_data = get_specific_month_data_to_predict(4)#assume that we choose April to predict the purchase decision of customers

#eda_purchase_count(purchase_data)

# purchase_data.drop(purchase_data.columns[2], axis=1, inplace=True)
# eda_purchase_and_average(purchase_data)

# purchase_data.drop(purchase_data.columns[[2, 5]], axis=1, inplace=True)
# corr = purchase_data.corr(method='pearson')
# print(np.asmatrix(corr))
# eda_purchase_var_correlation(corr)
# X = remove_correlative_feature(purchase_data, corr, 0.85)#0.85 as threshold of correlation to remove the feature
# X.to_csv("reduced_dataset.csv")

#random_forest_classifier_predict(purchase_data)

xgboost_classifier_predict(purchase_data)













