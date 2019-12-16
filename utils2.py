#! /usr/bin/env python3

# from utils2 import plot_pred, metric_report, CVscore_report,event_encoder,time_frame

import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, classification_report,get_scorer
from sklearn.model_selection import cross_val_score, KFold
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
sns.set()
import warnings
warnings.filterwarnings("ignore")


def plot_pred(X,y_true,y_pred,thresh =0.5,res=1000000):
    X_plot = X.copy()
    X_plot['predicted'] = y_pred

    plt.figure(figsize=(25,10))
    (2*y_true.head(res)).plot()
    (X_plot['predicted'] > thresh).astype(np.int).head(res).plot()
    plt.title('Plot of Model Predictions vs Actuals')
    plt.legend()
    plt.show()

def plot_probab(model,X,y_true,res=1000000):
    y_pred = model.predict_proba(X)[:,1]
    X_plot = X.copy()
    X_plot['predicted'] = y_pred

    plt.figure(figsize=(25,10))
    (2*y_true.head(res)).plot()
    X_plot['predicted'].head(res).plot()
    plt.title('Plot of Model Prediction Probabilities vs Actuals')
    plt.legend()
    plt.show()
    
def metric_report(model, X, y_true,thresh=0.5, name = None, plot = True):
    y_pred = model.predict_proba(X)[:,1]
    print("=" * 15, name, "=" * 15)
    print(f'          ROC AUC score: {roc_auc_score(y_true,y_pred)}')
    print(f'Balanced Accuracy score: {balanced_accuracy_score(y_true,y_pred > thresh)}\n') 
    print('\tConfusion Matrix\n',confusion_matrix(y_true,y_pred > thresh))
    print('\n')
    print(f'\t{name} Classification Report\n',classification_report(y_true,y_pred > thresh))
    print("=" * len("=" * 15 + " " + str(name) + " " + "=" * 15) + '\n')
    plot_pred(X,y_true,y_pred,thresh,res=1000000)
    
def CVscore_report(mod, X, y, cv = 5, name = None, scoring = 'roc_auc', custom = False):
    scoresCV = np.empty(cv, dtype = 'f8')
    
    if custom:
        s = get_scorer(scoring)
        kf = KFold(n_splits=cv, random_state=None, shuffle=True)
        
        for cv_cnt, (trn_idx, tst_idx) in enumerate(kf.split(X)):
            mod.fit(X.iloc[trn_idx], y.iloc[trn_idx])
            scoresCV[cv_cnt] = s(mod, X.iloc[tst_idx], y.iloc[tst_idx])
    
    else:
        scoresCV = cross_val_score(mod, X, y, cv = cv, scoring = get_scorer(scoring))
    
    print("=" * 15, "CV", name, "=" * 15)
    print("CV Mean:", scoresCV.mean())
    print("CV Std:", scoresCV.std())
    
    
def event_encoder(Events_list,df,lag=0,lag2=0):
    Events_list_lag=[]
    for i,event in enumerate(Events_list):
        if i%2 ==0:
            Events_list_lag.append(event+datetime.timedelta(minutes=-lag))
        if i%2 > 0:
            Events_list_lag.append(event+datetime.timedelta(minutes=lag2))

    df['Target_Anomaly']=0
    for i in range(0,len(Events_list)//2):
        df.loc[Events_list_lag[i*2]:Events_list_lag[(i*2)+1],'Target_Anomaly']=1
    return df
    
    
def time_frame(model,df,Event,num,early=360,late=0,thresh=0.5):
    df2=df.copy()
    y_pred = model.predict_proba(df2)[:,1]
    y_pred_outcome = y_pred > thresh
    df2['y_pred']=y_pred_outcome
    
    Event_times=df2.index[(df2['y_pred']==True)&(df2.index > (pd.to_datetime(Event[0]))+datetime.timedelta(minutes= -early)) & (df2.index < pd.to_datetime(Event[1])+datetime.timedelta(minutes= late))].tolist()
    if pd.to_datetime(Event[0])>Event_times[0]:
        Early_delta = pd.to_datetime(Event[0])-Event_times[0]
        e_l='early'
    else:
        Early_delta = Event_times[0]-pd.to_datetime(Event[0])
        e_l='late'
    
    print(f'-------------------------- Event {num} --------------------------')
    print(f' Predicted Start Anomaly Time: {Event_times[0]}.')
    print(f'   Predicted End Anomaly Time: {Event_times[-1]}.' )
    print(f'            Detected {e_l} by: {Early_delta}\n')


def feature_selection(df,n=25,Target_included=False):
    Top_features = ['Var182','Var141','Var167','Var158','Var1','Var138','Var101','Var4','Var43','Var3',
                    'Var0','Var2','Var95','Var139','Var97','Var122','Var96','Var34','Var93','Var7','Var15',
                    'Var118','Var166','Var37','Var98','Var193','Var18','Var142','Var171','Var174','Var126',
                    'Var92','Var103','Var163','Var160','Var159','Var8','Var117','Var99','Var162','Var116',
                    'Var21','Var120','Var102','Var121','Var156','Var107','Var12','Var206','Var35','Var36',
                    'Var109','Var203','Var42','Var175','Var189','Var213','Var76','Var215','Var134','Var129',
                    'Var119','Var90','Var187','Var152','Var44','Var164','Var185','Var148','Var41','Var108',
                    'Var6','Var11','Var212','Var214','Var144','Var200','Var211','Var140','Var14','Var39',
                    'Var128','Var149','Var143','Var33','Var145','Var31','Var23','Var136','Var104','Var24',
                    'Var125','Var165','Var172','Var13','Var30','Var131','Var216','Var137','Var183']
    
    Target = ['Target']
    
    if ('Target' in df.columns) and (Target_included==True):
        Selected = Top_features[0:n]+Target
    else:
        Selected =Top_features[0:n]
    
    y_df = df['Target_Anomaly']
    X_df = df[Selected]
    return X_df, y_df

def add_outcome(model,X,thresh):
    X2=X.copy()
    y_proba = model.predict_proba(X2)[:,1]
    y_outcome = (y_proba >thresh).astype('int')
    X2['y_proba']  =y_proba
    X2['y_outcome']=y_outcome
    return X2

def return_outcome(df,timestamp):
    a=df[df.index==pd.to_datetime(timestamp)]['y_outcome']
    return a
    