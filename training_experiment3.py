#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from decimal import Decimal
from datetime import datetime


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score


# In[ ]:


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


# In[ ]:


domain_bucket = {
    "_01" : Interval(0, 1, upper_closed=False),
    "_02" : Interval(1, 5, upper_closed=False),
    "_03" : Interval(5, 20, upper_closed=False),
    "_04" : Interval(20, 50, upper_closed=False),
    "_05" : Interval(50, 100, upper_closed=True),}


# In[ ]:


with open("./Encode_file/cate_onehot_list.pkl", 'rb') as f:
    domain_cate = pickle.load(f)
with open("./Encode_file/rank_cate.pkl", 'rb') as f:
    rank = pickle.load(f)


# In[ ]:


### default is predict top one domain
cs_rank_list = "CS_{}".format(rank[0])
ecs_rank_list = "ECS_{}".format(rank[0])


# In[ ]:


### first merge each UUID data with day unit


# In[ ]:


paths, tags = get_file_list("./cs_shift_result/")
c_paths, c_tags = get_file_list("./csall_shift_result/")
e_paths, e_tags = get_file_list("./csecs_shift_result/")


# In[ ]:


try:
    rmt("./_days_cs")
    rmt("./_days_cs_all")
    rmt("./_days_cs_ecs")
except:
    pass
finally:
    os.mkdir("./_days_cs")
    os.mkdir("./_days_cs_all")
    os.mkdir("./_days_cs_ecs")


# In[ ]:


def do_date_join(date_uni, path, dataset): 
    mer = pd.read_csv(path)
    domain = []
    total = []
    for i in date_uni:
        cate = mer.loc[mer.date == i]['cate'].reset_index(drop=True)
        e_type = mer.loc[mer.date == i]['type_encode'].reset_index(drop=True)
        count_cs = np.zeros(len(domain_cate)) 
        count_ecs = np.zeros(len(domain_cate))
        for j in range(len(cate)): 
            if e_type[j] in range(11):
                count_cs[domain_cate.tolist().index(cate[j])] += 1
            else:
                count_ecs[domain_cate.tolist().index(cate[j])] += 1
                
        total.append(count_cs.sum() + count_ecs.sum())
        count_cs = _decimal(count_cs / count_cs.sum())
        if dataset == 'ecs':
            count_ecs = _decimal(count_ecs / count_ecs.sum())
        else:
            count_ecs = _decimal(count_ecs)     
        domain.append(np.hstack((count_cs,count_ecs)))
    
    new_columns = np.array([x.format(c) for x in ["CS_{}", "ECS_{}"] for c in domain_cate.tolist()]) 
    domain_df = pd.DataFrame(domain, columns = new_columns)

    total_df = pd.DataFrame(total, columns = ['total'])
    new_mer = domain_df.join(total_df, how='left')
    return new_mer


# In[ ]:


def _decimal(array):
    for i in range(len(array)):
        array[i] = Decimal(array[i]).quantize(Decimal('0.000'))
    return array


# In[ ]:


def _join_data(paths, tags, p, dataset):
    
    for i in tqdm(range (len(paths))):
        date_uni = pd.read_csv(paths[i])['date'].unique()
        date_df = pd.DataFrame(date_uni, columns = ['date'])
        new_df = do_date_join(date_uni, paths[i], dataset)
        new_df = date_df.join(new_df, how='left')
        new_df.to_csv("{}/{}.csv".format(p ,tags[i]), index = False)


# In[ ]:


_join_data(paths, tags, './_days_cs', 'cs')
_join_data(c_paths, c_tags, './_days_cs_all', 'cs')
_join_data(e_paths, e_tags, './_days_cs_ecs', 'ecs')


# In[ ]:


### split train/test/eval


# In[ ]:


def train_test_split(args):
    
    p, tag, to_path = args
    mer = pd.read_csv(p)
   
    if mer.shape[0] == 0:
        print(tag[:10], "data error")
        return
    if mer.shape[0] >= 30:
        test_part = mer.loc[mer.shape[0]-14::]
        remain_part = mer.loc[0:mer.shape[0]-15]
    else:
        mer.to_csv("{}/train/{}.csv".format(to_path, tag), index=False)
        print(tag[:10], "data only use for train")
        return
    
    if remain_part.shape[0] >= 60:
        remain_part.loc[remain_part.shape[0]-14::].reset_index(drop=True).to_csv("{}/eval/{}.csv".format(to_path, tag), index=False)
        remain_part.loc[0:remain_part.shape[0]-15].reset_index(drop=True).to_csv("{}/train/{}.csv".format(to_path, tag), index=False)
    else:
        remain_part.reset_index(drop=True).to_csv("{}/train/{}.csv".format(to_path, tag), index=False)
    
    test_part.to_csv("{}/test/{}.csv".format(to_path, tag), index=False)
    return


# In[ ]:


def do_train_test_split(work_dir):
    print("=== Train Test Split ===")
    
    tmp = work_dir.split('/')
    tmp[1] = tmp[1] + "_trevte"
    to_path = "/".join(tmp)
    paths, tags = get_file_list(work_dir)
    try:
        rmt(to_path)
    except:
        pass
    finally:
        os.mkdir(to_path)
        os.mkdir("{}/train".format(to_path) )
        os.mkdir("{}/eval".format(to_path) )
        os.mkdir("{}/test".format(to_path) )
    
    print("Working on dir: {}\nSave result to: {}".format(work_dir, to_path))
 
    for i, p in tqdm(enumerate(paths)):
        train_test_split([p, tags[i], to_path])
    return


# In[ ]:


do_train_test_split("./_days_cs/")
do_train_test_split("./_days_cs_all/")
do_train_test_split("./_days_cs_ecs/")


# In[ ]:


db = np.array(['_01', '_02', '_03','_04', '_05'])


# In[ ]:


def _bucket(values):
    for i in range(len(values)):
        for j in domain_bucket.keys():
            if values[i] in domain_bucket[j]:
                values[i] = db.tolist().index(j)
    return values

def __bucket(value):
    for j in domain_bucket.keys():
        if value in domain_bucket[j]:
            value = db.tolist().index(j)
    return value


# In[ ]:


def create_feature(paths):
    for i in tqdm(range (len(paths))):
        mer = pd.read_csv(paths[i])
        count_cs = []
        count_ecs = []
        for j in range(len(mer)):
            count_cs.append(np.count_nonzero(mer.loc[j, col_cs]))
            count_ecs.append(np.count_nonzero(mer.loc[j, col_ecs]))
            
        feature_1 = pd.DataFrame(count_cs, columns = ['cs_domain'])
        feature_2 = pd.DataFrame(count_ecs, columns = ['ecs_domain'])
        
        mer = mer.join(feature_1, how='left')
        new_mer = mer.join(feature_2, how='left')
        
        new_mer.to_csv(paths[i], index = False)


# In[ ]:


def create_dataset(paths, num):
    target=[]
    label=[]
    for i in tqdm(range (len(paths))):
        mer = pd.read_csv(paths[i])
        for i in range(0+num, len(mer)):
            c = []
            label.append(__bucket(mer.loc[i, cs_rank_list]* 100))
            for j in range(i-num, i):
                v = []
                v.append(__bucket(mer.loc[j, cs_rank_list] * 100))
                v.append(__bucket(mer.loc[j, ecs_rank_list] * 100))
                v.append(mer.loc[j, 'cs_domain'])
                v.append(mer.loc[j, 'ecs_domain'])
                v.append(mer.loc[j, 'total'])
                c.append(np.array(v))
            target.append(c)                   
    return np.array(target), np.array(label)


# In[ ]:


paths_train, tags_train = get_file_list("./_days_cs_trevte/train/")
paths_test, tags_test = get_file_list("./_days_cs_trevte/test/")
paths_eval, tags_eval = get_file_list("./_days_cs_trevte/eval/")

c_paths_train, c_tags_train = get_file_list("./_days_cs_all_trevte/train/")
c_paths_test, c_tags_test = get_file_list("./_days_cs_all_trevte/test/")
c_paths_eval, c_tags_eval = get_file_list("./_days_cs_all_trevte/eval/")

e_paths_train, e_tags_train = get_file_list("./_days_cs_ecs_trevte/train/")
e_paths_test, e_tags_test = get_file_list("./_days_cs_ecs_trevte/test/")
e_paths_eval, e_tags_eval = get_file_list("./_days_cs_ecs_trevte/eval/")


# In[ ]:


col_cs = pd.read_csv(paths_train[0]).columns[1:83]
col_ecs = pd.read_csv(paths_train[0]).columns[83:165]


# In[ ]:


create_feature(paths_train)
create_feature(paths_test)
create_feature(paths_eval)

create_feature(c_paths_train)
create_feature(c_paths_test)
create_feature(c_paths_eval)

create_feature(e_paths_train)
create_feature(e_paths_test)
create_feature(e_paths_eval)


# In[ ]:


five_target_train, five_label_train = create_dataset(paths_train, 5)
five_target_test, five_label_test = create_dataset(paths_test, 5)
five_target_eval, five_label_eval = create_dataset(paths_eval, 5)

c_five_target_train, c_five_label_train = create_dataset(c_paths_train, 5)
c_five_target_test, c_five_label_test = create_dataset(c_paths_test, 5)
c_five_target_eval, c_five_label_eval = create_dataset(c_paths_eval, 5)

e_five_target_train, e_five_label_train = create_dataset(e_paths_train, 5)
e_five_target_test, e_five_label_test = create_dataset(e_paths_test, 5)
e_five_target_eval, e_five_label_eval = create_dataset(e_paths_eval, 5)


# In[ ]:


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


five_label_train = np.array(five_label_train, dtype=int)
five_label_test = np.array(five_label_test, dtype=int)
five_label_eval = np.array(five_label_eval, dtype=int)

c_five_label_train = np.array(c_five_label_train, dtype=int)
c_five_label_test = np.array(c_five_label_test, dtype=int)
c_five_label_eval = np.array(c_five_label_eval, dtype=int)

e_five_label_train = np.array(e_five_label_train, dtype=int)
e_five_label_test = np.array(e_five_label_test, dtype=int)
e_five_label_eval = np.array(e_five_label_eval, dtype=int)


# In[ ]:


def RF(target, label, t_target, t_label, d):   
    clf = RandomForestClassifier(n_estimators=100, n_jobs = -1, max_depth = d).fit(target, label.ravel())
    pre_test = clf.predict(t_target)
    f1 = f1_score(t_label, pre_test, average='micro')
    matrix = confusion_matrix(y_true = t_label, y_pred = pre_test)
    del clf
    print(f1)
    
def knn(target, label, t_target, t_label, n):
    clf = KNeighborsClassifier(n_neighbors = n ,n_jobs = -1).fit(target, label.ravel())
    pre_test = clf.predict(t_target)
    f1 = f1_score(t_label, pre_test, average='micro')
    matrix = confusion_matrix(y_true = t_label, y_pred = pre_test)
    del clf
    print(f1)
    
def xgb(target, label, t_target, t_label, d):
    clf = XGBClassifier(learning_rate=.1, n_estimators=100, max_depth = d).fit(target, label.ravel())
    pre_test = clf.predict(t_target)
    f1 = f1_score(t_label, pre_test, average='micro')
    matrix = confusion_matrix(y_true = t_label, y_pred = pre_test)
    del clf
    print(f1)
    
def log(target, label, t_target, t_label, c):
    clf = LogisticRegression(solver = 'saga', max_iter = 20000,
                             multi_class = 'multinomial', C = c).fit(target, label.ravel())
    pre_test = clf.predict(t_target)
    f1 = f1_score(t_label, pre_test, average='micro')
    matrix = confusion_matrix(y_true = t_label, y_pred = pre_test)
    del clf
    print(f1)


# In[ ]:


RF(five_target_train, five_label_train, five_target_test, five_label_test, 6)
RF(c_five_target_train, c_five_label_train, c_five_target_test, c_five_label_test, 10)
RF(e_five_target_train, e_five_label_train, e_five_target_test, e_five_label_test, 14)

knn(five_target_train, five_label_train, five_target_test, five_label_test, 39)
knn(c_five_target_train, c_five_label_train, c_five_target_test, c_five_label_test, 36)
knn(e_five_target_train, e_five_label_train, e_five_target_test, e_five_label_test, 27)

log(five_target_train, five_label_train, five_target_test, five_label_test, 1)
log(c_five_target_train, c_five_label_train, c_five_target_test, c_five_label_test, 1)
log(e_five_target_train, e_five_label_train, e_five_target_test, e_five_label_test, 1)

xgb(five_target_train, five_label_train, five_target_test, five_label_test, 3)
xgb(c_five_target_train, c_five_label_train, c_five_target_test, c_five_label_test, 2)
xgb(e_five_target_train, e_five_label_train, e_five_target_test, e_five_label_test, 3)


# In[ ]:




