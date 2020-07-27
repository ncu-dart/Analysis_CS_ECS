#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import swifter
import numpy as np
import os
from shutil import rmtree as rmt
from interval import Interval
import pickle
from tqdm.notebook import tqdm


# In[ ]:


# final_focus_all.parquet (ecs data)
# final_history_all.parquet (cs data)
# final_history_new.parquet (ics data)
# CS_ECS_new.parquet (ics + ecs data) 


# In[2]:


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


# In[3]:


pq_history_all = pd.read_parquet('Pre-processing/final_history_all.parquet')
pq_focus = pd.read_parquet('Pre-processing/final_focus_all.parquet')
pq_history = pd.read_parquet('Pre-processing/final_history_new.parquet')
pq_CS_ECS = pd.concat([pq_history, pq_focus], sort=False).sort_values(by=['UUID', 'dt']).reset_index(drop=True)
pq_CS_ECS.to_parquet('Pre-processing/CS_ECS_new.parquet')


# In[4]:


def do_dt_shift(df, uids):
    # check sharing tmp folder
    try:
        rmt("./_tmp(all)")
    except:
        pass
    finally:
        os.mkdir("./_tmp(all)")


    print("Start dt shift...")

    for uid in tqdm(uids):
        dt_shift(df[df['UUID'] == uid].copy())

    print("Merging Result...")

    paths, tags = get_file_list("./_tmp(all)")

    print("DT Shifted, store in merged")

    return paths


# In[5]:


def dt_shift(df):
    # display(df)
    last_idx = df.index[0]
    active_dt = 0
    for i in df.index:
        if df.loc[i, 'type'] == 'idle':
            last_idx = i
        elif df.loc[i, 'type'] == 'active':
            active_dt = int(df.loc[i, 'dt'])
            for idx in range(last_idx + 1, i):
                if int(df.loc[idx, 'dt'])in Interval(active_dt - 1000, active_dt, lower_closed=False):
                    df.loc[idx, 'dt'] = active_dt+1     
            last_idx = i
    
    df.to_csv("./_tmp(all)/{}.csv".format(df.loc[df.index[0], 'UUID']), index=False )
    return 0


# In[6]:


cs_all_uid = pq_history_all['UUID'].unique()
cs_ecs_uid = pq_CS_ECS['UUID'].unique()
cs_uid = pq_history['UUID'].unique()

cs_paths = do_dt_shift(pq_history, cs_uid)
pq_history = pd.concat([pd.read_csv(p) for p in cs_paths]).sort_values(by=['UUID', 'dt']).reset_index(drop=True)
cs_all_paths = do_dt_shift(pq_history_all, cs_all_uid)
pq_CS_all = pd.concat([pd.read_csv(p) for p in cs_all_paths]).sort_values(by=['UUID', 'dt']).reset_index(drop=True)
cs_ecs_paths = do_dt_shift(pq_CS_ECS, cs_ecs_uid)
pq_CS_ECS = pd.concat([pd.read_csv(p) for p in cs_ecs_paths]).sort_values(by=['UUID', 'dt']).reset_index(drop=True)


# In[ ]:


# save the data file which create above
# pq_CS_ECS.to_parquet('Pre-processing/shift_CS_ECS_new.parquet')
# pq_CS_all.to_parquet('Pre-processing/shift_CS_all.parquet')
# pq_history.to_parquet('Pre-processing/shift_CS.parquet')


# In[9]:


pq_CS_all = pq_CS_all[['UUID', 'cate', 'date', 'datetime', 'domain', 'dt','type']]
pq_history = pq_history[['UUID', 'cate', 'date', 'datetime', 'domain', 'dt', 'type']]
pq_CS_ECS = pq_CS_ECS[['UUID', 'cate', 'date', 'datetime', 'domain', 'dt', 'type']]


# In[11]:


try:
    rmt("./csall_shift_result")
    rmt("./cs_shift_result")
    rmt("./csecs_shift_result")
except:
    pass
finally:
    os.mkdir("./csall_shift_result")
    os.mkdir("./cs_shift_result")
    os.mkdir("./csecs_shift_result")


# In[26]:


time_bucket = {
    "L00_5S" : Interval(0, 5 * 1e3, upper_closed=False),
    "L01_20S": Interval(5 * 1e3, 20 * 1e3, upper_closed=False),
    "L02_2M" : Interval(20 * 1e3, 2 * 60 * 1e3, upper_closed=False),
    "L03_20M": Interval(2 * 60 * 1e3, 20 * 60 * 1e3, upper_closed=False),
    "L04_NW" : Interval(20 * 60 * 1e3, np.inf),}

with open("./Encode_file/event_onehot_list.pkl", 'rb') as f:
    event_cate = pickle.load(f)
with open("./Encode_file/event_all_onehot_list.pkl", 'rb') as f:
    event_all_cate = pickle.load(f)
with open("./Encode_file/cate_onehot_list.pkl", 'rb') as f:
    domain_cate = pickle.load(f)

TB = np.array([x.format(tl) for x in ["ECS,{}", "CS,{}"] for tl in time_bucket.keys()])


# In[29]:


def add_columns(mer):
 
    mer['time'] = (mer['dt'] - mer['dt'].shift(1)).swifter.apply(lambda x : x)
    mer['bucket'] = mer['time'].swifter.apply(lambda x : [k for k in time_bucket.keys() if x in time_bucket[k]])
    mer['last'] = mer['type'].shift(1).swifter.apply(lambda x : [ 'CS' if x in event_all_cate[0:11] else 'ECS'])
    mer['time_bucket'] = (mer['last']+mer['bucket']).swifter.apply(lambda x : x)
    mer['time_bucket'] = mer['time_bucket'].swifter.apply(lambda x:str(x))
    
    mer['time_bucket'] = mer['time_bucket'].swifter.apply(
        lambda x: x.replace("'", "").replace("[", "").replace("]", "").replace(" ", ""))
    
    mer = mer.drop(['bucket', 'last'], axis=1).reset_index(drop=True)
    mer = mer.drop([0]).reset_index(drop=True)

    mer['cate_encode'] = mer['cate'].swifter.apply(lambda x: domain_cate.tolist().index(x))
    mer['type_encode'] = mer['type'].swifter.apply(lambda x: event_all_cate.tolist().index(x))
    mer['tb_encode'] = mer['time_bucket'].swifter.apply(lambda x: TB.tolist().index(x))

    return mer.reset_index(drop=True)


# In[30]:


for uid in tqdm(cs_all_uid):
    mer = pq_CS_all[pq_CS_all['UUID'] == uid].copy()
    if len(mer) > 15:
        mer = add_columns(mer)      
        mer.to_csv('./csall_shift_result/{}.csv'.format(uid), index = False)
        
for uid in tqdm(cs_uid):
    mer = pq_history[pq_history['UUID'] == uid].copy()
    if len(mer) > 15:
        mer = add_columns(mer)
        mer.to_csv('./cs_shift_result/{}.csv'.format(uid), index = False)
        
for uid in tqdm(cs_ecs_uid):
    mer = pq_CS_ECS[pq_CS_ECS['UUID'] == uid].copy()
    if len(mer) > 15:
        mer = add_columns(mer) 
        mer.to_csv('./csecs_shift_result/{}.csv'.format(uid), index = False)

### split data
# In[31]:


def train_test_split(args, file_type, uid2, uid3, max_dt2, max_dt3):
    p, tag, to_path = args
    maxdt2= 0
    maxdt3 = 0
    
    d = pd.read_csv(p)
    
    if d.shape[0] == 0:
        print(tag[:10], "test error")
        return uid2, uid3, max_dt2, max_dt3
    
    if file_type == 'CS':
        maxdt2 = max(d.dt)
        max_dt2.append(max(d.dt))
        uid2.append(d['UUID'][d.loc[d.dt == max(d.dt)].index.tolist()[0]])
    else:
        maxdt2 = max_dt2[uid2.index(tag)]
    
    test_part = d.loc[d.dt - (maxdt2 -  5 * 86400 * 1000) >= 0 ].reset_index(drop=True)
    remain_part = d.loc[d.dt - (maxdt2 -  5 * 86400 * 1000) < 0 ]

    if remain_part.shape[0] == 0:
        d.to_csv("{}/train/{}.csv".format(to_path, tag), index=False)
        print(tag[:10], "remain error")
        return uid2, uid3, max_dt2, max_dt3
    
    else:
        remain_days = (max(remain_part.dt) - min(remain_part.dt) ) / 86400000
        if remain_days >= 25:
            if file_type == 'CS':
                maxdt3 = max(remain_part.dt)
                max_dt3.append(max(remain_part.dt))
                uid3.append(d['UUID'][d.loc[d.dt == max(remain_part.dt)].index.tolist()[0]])
            else:
                if tag not in uid3:
                    print('===this uuid with cs doesnt have eval, please check!!!===')
                    print(tag)
                    return uid2, uid3, max_dt2, max_dt3 
                else:
                    maxdt3 = max_dt3[uid3.index(tag)]
            
            remain_part.loc[remain_part.dt - (maxdt3 - 5 * 86400 * 1000) >= 0 ].reset_index(drop=True).to_csv("{}/eval/{}.csv".format(to_path, tag ) , index=False)
            remain_part.loc[remain_part.dt - (maxdt3 - 5 * 86400 * 1000) < 0 ].reset_index(drop=True).to_csv("{}/train/{}.csv".format(to_path, tag ) , index=False)
        else:
            remain_part.reset_index(drop=True).to_csv("{}/train/{}.csv".format(to_path, tag ) , index=False)

        test_part.to_csv("{}/test/{}.csv".format(to_path, tag ), index=False )
        
    return uid2, uid3, max_dt2, max_dt3


# In[32]:


def do_train_test_split(work_dir, filetype, uid2, uid3, max_dt2, max_dt3):
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
    #print(paths)
    for i, p in tqdm(enumerate(paths)):
        
        uid2, uid3, max_dt2, max_dt3 = train_test_split([p, tags[i], to_path], filetype, uid2, uid3, max_dt2, max_dt3)
        
    return uid2, uid3, max_dt2, max_dt3


# In[33]:


uid1 = []
max_dt2 = []
uid2 = []
max_dt3 = []
uid3 =[]


# In[34]:


uid2, uid3, max_dt2, max_dt3 = do_train_test_split("./cs_shift_result/", 'CS', uid2, uid3, max_dt2, max_dt3)
uid2, uid3, max_dt2, max_dt3 = do_train_test_split("./csall_shift_result/", 'CS_All', uid2, uid3, max_dt2, max_dt3)
uid2, uid3, max_dt2, max_dt3 = do_train_test_split("./csecs_shift_result/", 'CS_ECS', uid2, uid3, max_dt2, max_dt3)


# In[35]:


get_ipython().system('jupyter nbconvert — to script data_processing.ipynb')


# In[ ]:




