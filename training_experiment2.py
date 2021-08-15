#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from shutil import rmtree as rmt
from interval import Interval
import os 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import pickle
import swifter
import torch 
from sklearn.preprocessing import OneHotEncoder 
import collections
from sklearn.utils import shuffle
from decimal import Decimal
from sklearn.metrics import f1_score
from collections import Counter


# In[2]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn import metrics


# In[3]:


def get_file_list(root, ftype = ".csv"):
    FileList = []
    filename = []
    for dirPath, dirNames, fileNames in os.walk(root):
        for f in fileNames:
            if f.find(ftype) > -1:
                FileList.append(os.path.join(dirPath, f))
                filename.append(f.replace(ftype, ""))
    if len(filename) > 0:
        a = zip(FileList, filename)
        a = sorted(a, key = lambda t : t[1])
        FileList, filename = zip(*a)
    return FileList, filename


# In[4]:


time_bucket = {
    "L00_5S" : Interval(0, 5 * 1e3, upper_closed=False),
    "L01_20S": Interval(5 * 1e3, 20 * 1e3, upper_closed=False),
    "L02_2M" : Interval(20 * 1e3, 2 * 60 * 1e3, upper_closed=False),
    "L03_20M": Interval(2 * 60 * 1e3, 20 * 60 * 1e3, upper_closed=False),
    "L04_NW" : Interval(20 * 60 * 1e3, np.inf),}
tb = np.array([x.format(tl) for x in ["{}"] for tl in time_bucket.keys()])


# In[5]:


def create_dataset(paths, domain_cate, num):
    target=[]
    label=[]
    for i in tqdm(range (len(paths))):
        mer = pd.read_csv(paths[i])
        for i in range(0+num, len(mer)):
            c = []            
            if mer.loc[i, 'type_encode'] in range(11):
                #label.append(mer.loc[i ,'cate_encode'])
                label.append(tb.tolist().index(mer.loc[i ,'time_bucket'].split(',')[1]))
                for j in range(i-num, i):
                    b=[]
                    b.append(mer.loc[j, 'cate_encode'])
                    b.append(mer.loc[j, 'type_encode'])
                    b.append(mer.loc[j, 'tb_encode'])
                    c.append(b)
                target.append(c)                   
    return np.array(target), np.array(label)


# In[6]:


def ecs_create_dataset(paths, domain_cate, count):
    target=[]
    label=[]
    for i in tqdm( range (len(paths)) ):
        mer = pd.read_csv(paths[i])
        for i in range(0+count, len(mer)):
            c = []            
            if mer.loc[i, 'type_encode'] in range(7):
                #label.append(mer.loc[i ,'cate_encode'])
                label.append(tb.tolist().index(mer.loc[i ,'time_bucket'].split(',')[1]))
                for j in range(i-count, i):
                    b=[]
                    b.append(mer.loc[j, 'cate_encode'])
                    b.append(mer.loc[j, 'type_encode'])
                    b.append(mer.loc[j, 'tb_encode'])
                    c.append(b)
                target.append(c)                   
    return np.array(target), np.array(label)


# In[7]:


with open("./Encode_file/cate_onehot_list.pkl", 'rb') as f:
    domain_cate = pickle.load(f)
with open("./Encode_file/event_onehot_list.pkl", 'rb') as f:
    event_type = pickle.load(f)
with open("./Encode_file/event_all_onehot_list.pkl", 'rb') as f:
    event_all_type = pickle.load(f)


# In[ ]:


### to get the file data split before


# In[8]:


paths_train, tags_train = get_file_list("./cs_shift_result_trevte/train/")
paths_test, tags_test = get_file_list("./cs_shift_result_trevte/test/")
paths_eval, tags_eval = get_file_list("./cs_shift_result_trevte/eval/")

c_paths_train, c_tags_train = get_file_list("./csall_shift_result_trevte/train/")
c_paths_test, c_tags_test = get_file_list("./csall_shift_result_trevte/test/")
c_paths_eval, c_tags_eval = get_file_list("./csall_shift_result_trevte/eval/")

e_paths_train, e_tags_train = get_file_list("./csecs_shift_result_trevte/train/")
e_paths_test, e_tags_test = get_file_list("./csecs_shift_result_trevte/test/")
e_paths_eval, e_tags_eval = get_file_list("./csecs_shift_result_trevte/eval/")


# In[ ]:


### input data create (use five event datas to predict next event)


# In[9]:


five_target_train, five_label_train = create_dataset(paths_train, domain_cate, 5)
five_target_test, five_label_test = create_dataset(paths_test, domain_cate, 5)
five_target_eval, five_label_eval = create_dataset(paths_eval, domain_cate, 5)

c_five_target_train, c_five_label_train = create_dataset(c_paths_train, domain_cate, 5)
c_five_target_test, c_five_label_test = create_dataset(c_paths_test, domain_cate, 5)
c_five_target_eval, c_five_label_eval = create_dataset(c_paths_eval, domain_cate, 5)

e_five_target_train, e_five_label_train = ecs_create_dataset(e_paths_train, domain_cate, 5)
e_five_target_test, e_five_label_test = ecs_create_dataset(e_paths_test, domain_cate, 5)
e_five_target_eval, e_five_label_eval = ecs_create_dataset(e_paths_eval, domain_cate, 5)


# In[ ]:


### reshape size to let them fit sklearn model input shape


# In[10]:


five_target_train = five_target_train.reshape(five_target_train.shape[0],-1)
five_target_test = five_target_test.reshape(five_target_test.shape[0],-1)
five_target_eval = five_target_eval.reshape(five_target_eval.shape[0],-1)

c_five_target_train = c_five_target_train.reshape(c_five_target_train.shape[0],-1)
c_five_target_test = c_five_target_test.reshape(c_five_target_test.shape[0],-1)
c_five_target_eval = c_five_target_eval.reshape(c_five_target_eval.shape[0],-1)

e_five_target_train = e_five_target_train.reshape(e_five_target_train.shape[0],-1)
e_five_target_test = e_five_target_test.reshape(e_five_target_test.shape[0],-1)
e_five_target_eval = e_five_target_eval.reshape(e_five_target_eval.shape[0],-1)


# In[ ]:


### model use in experiment one 
### (you might find best parameters from each model first using eval data)
### (below is using parameters which i found before you can change by yourself)


# In[11]:


def random_forest_acc(target_train, label_train, target_test, label_test, d):     
    clf = RandomForestClassifier(n_estimators=100,n_jobs = -1, max_depth = d).fit(target_train, label_train)
    pred = clf.predict(target_test)
    f1 = f1_score(label_test, pred, average='micro')
    del clf
    print("Testing score:%f"%(f1))
    print("--------------------------------") 
    return pred

def knn(target_train, label_train, target_test, label_test, n):
    clf = KNeighborsClassifier(n_neighbors = n, n_jobs = -1).fit(target_train, label_train)
    pred = clf.predict(target_test)
    f1 = f1_score(label_test, pred, average='micro')
    del clf   
    print("Testing score:%f"%(f1))
    print("--------------------------------") 
    return pred

def xgb_acc(target_train, label_train, target_test, label_test, d):
    clf = XGBClassifier(learning_rate=.1, n_estimators=100, max_depth=d).fit(target_train, label_train)
    pred = clf.predict(target_test)
    f1 = f1_score(label_test, pred, average='micro')
    del clf   
    print("Testing score:%f"%(f1))
    print("--------------------------------") 
    return pred

def log_acc(target_train, label_train, target_test, label_test):
    clf = LogisticRegression(solver = 'sag', max_iter = 20000, C = 1,
                             multi_class = 'multinomial').fit(target_train, label_train)
    pred = clf.predict(target_test)
    f1 = f1_score(label_test, pred, average='micro')
    del clf
    print("Testing score:%f"%(f1))
    print("--------------------------------") 
    return pred


# In[12]:


rf_ics_p = random_forest_acc(five_target_train, five_label_train, five_target_test, five_label_test, 22)
rf_cs_p = random_forest_acc(c_five_target_train, c_five_label_train, c_five_target_test, c_five_label_test, 22)
rf_csecs_p = random_forest_acc(e_five_target_train, e_five_label_train, e_five_target_test, e_five_label_test, 24)

knn_ics_p = knn(five_target_train, five_label_train, five_target_test, five_label_test, 18)
knn_cs_p = knn(c_five_target_train, c_five_label_train, c_five_target_test, c_five_label_test, 21)
knn_csecs_p = knn(e_five_target_train, e_five_label_train, e_five_target_test, e_five_label_test, 15)

xgb_ics_p = xgb_acc(five_target_train, five_label_train, five_target_test, five_label_test, 11)
xgb_cs_p = xgb_acc(c_five_target_train, c_five_label_train, c_five_target_test, c_five_label_test, 12)
xgb_csecs_p = xgb_acc(e_five_target_train, e_five_label_train, e_five_target_test, e_five_label_test, 12)

log_ics_p = log_acc(five_target_train, five_label_train, five_target_test, five_label_test)
log_cs_p = log_acc(c_five_target_train, c_five_label_train, c_five_target_test, c_five_label_test)
log_csecs_p = log_acc(e_five_target_train, e_five_label_train, e_five_target_test, e_five_label_test)


# In[ ]:


### get each model pred results can use them to get confusion matrix 


# In[ ]:


def multi_result(target_test, label_test, num):
    file = pd.DataFrame({'true/predict':list(range(83))})
    
    for i in range(num):
        y_cal = calculate_result(target_test, label_test, i, num)
        df = pd.DataFrame.from_dict(y_cal, orient="index", columns = [str(i)])
        file = file.join(df, on = 'true/predict', how='left').fillna(0).astype('int32')
    return file.set_index('true/predict')
        
def calculate_result(result, label, c, num):
    cal = []
    for i in range(len(result)):
        if result[i] == c:
            cal.append(label[i])
    cal = collections.Counter(cal)
    y_cal = dict(sorted(cal.items(),key = lambda i: i[0]))
    return y_cal


# In[ ]:


def table(label, pred):
    label_count = Counter(label)
    ### all * all
    file = multi_result(pred, label, 82)
    ### 20 * 20
    file1 = file[(str(i) for i in rank_encode)].loc[rank_encode]
    ###
    new = file1.copy().astype('str')
    k = -1
    for i in new.index:
        k += 1
        for j in range(len(file1.columns)):
            c = str(int(file1.loc[i][j])/label_count[rank_encode[k]]*100)
            new.loc[i][j] = (str(new.loc[i][j]) + '(' + str(Decimal(c).quantize(Decimal('0.00'))) + ')')
    ### 
    top['i'] = top['true/predict'].swifter.apply(lambda x : domain_cate.tolist().index(x))
    ttop = top.set_index('i')
    ###
    file2 = new.join(ttop, how='left')
    file2 = file2.set_index('true/predict').rename(columns=lambda x: domain_cate[int(x)])
    ###
    d = pd.DataFrame({'true/predict':list(rank[0:20])})
    d = d.join(pd.DataFrame({'true / predict':list(rank[0:20])}), how='left').set_index('true / predict')
    file2 = d.join(file2, how='left')
    return file2


# In[ ]:


with open("./Encode_file/rank_cate.pkl", 'rb') as f:
    rank = pickle.load(f)
#top = pd.DataFrame({'true/predict':Flist(rank[0:20])})

rank_encode= []
for i in range(20):
    rank_encode.append(domain_cate.tolist().index(rank[i]))


# In[ ]:


### table of experiment one results (table only get top20 domain type)
### if you want to create experiment two table do it by yourself


# In[ ]:


#rf_ics_f = table(five_label_test, rf_ics_p)
#rf_cs_f = table(c_five_label_test, rf_cs_p)
#rf_csecs_f = table(e_five_label_test, rf_csecs_p)
#
#knn_ics_f = table(five_label_test, knn_ics_p)
#knn_cs_f = table(c_five_label_test, knn_cs_p)
#knn_csecs_f = table(e_five_label_test, knn_csecs_p)
#
#xgb_ics_f = table(five_label_test, xgb_ics_p)
#xgb_cs_f = table(c_five_label_test, xgb_cs_p)
#xgb_csecs_f = table(e_five_label_test, xgb_csecs_p)
#
#log_ics_f = table(five_label_test, log_ics_p)
#log_cs_f = table(c_five_label_test, log_cs_p)
#log_csecs_f = table(e_five_label_test, log_csecs_p)


# In[ ]:


#rf_ics_f.to_csv('rf_ics.csv', index = False)
#rf_cs_f.to_csv('rf_cs.csv', index = False)
#rf_csecs_f.to_csv('rf_icsecs.csv', index = False)
#
#knn_ics_f.to_csv('knn_ics.csv', index = False)
#knn_cs_f.to_csv('knn_cs.csv', index = False)
#knn_csecs_f.to_csv('knn_icsecs.csv', index = False)
#
#xgb_ics_f.to_csv('xgb_ics.csv', index = False)
#xgb_cs_f.to_csv('xgb_cs.csv', index = False)
#xgb_csecs_f.to_csv('xgb_icsecs.csv', index = False)
#
#log_ics_f.to_csv('log_ics.csv', index = False)
#log_cs_f.to_csv('log_cs.csv', index = False)
#log_csecs_f.to_csv('log_icsecs.csv', index = False)


# In[ ]:




