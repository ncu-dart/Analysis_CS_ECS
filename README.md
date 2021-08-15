# Analysis_CS_ECS

## Summary of running the code

1. Create a folder ```Pre-processing``` in the root folder of the project.
1. Run ```python3 data_processin.py``` to generate training/validation/test datasets.
1. Run ```python3 training_experiment1_2.py``` to conduct experiment 1.
1. Run ```python3 training_experiment2.py``` to conduct experiment 2.
1. Run ```python3 training_experiment3.py``` to conduct experiment 3.


## data_processing.py

- .parquet : 檔案下載連結 https://drive.google.com/drive/folders/1IeYoKJltNklte72Hk7kOJSd2x-WCMbrf?usp=sharing

- shift_.parquet : 調整過事件時間順序之資料──前處理之時間單位  

- shift_result_ :  

  1. 加入新的欄位 ── time_bucket/cate encode/type encode/tb encode  
  2. 去除使用量較少的使用者 ── <15 events
  3. 依照UUID分類存成.csv
  
- shift_result_trevte : 將上面的資料依天數切成train/eval/test

## training_experiment1_2.py

- 直接運行得到的是實驗一之結果

- 要做實驗二  

  1. 請註解掉  
     create_dataset/ecs_create_dataset中 "label.append(mer.loc[i ,'cate_encode'])"  
  2. 去掉註解  
     create_dataset/ecs_create_dataset中 "label.append(tb.tolist().index(mer.loc[i ,'time_bucket'].split(',')[1]))"  
  3. 原.py中.csv生成得table是配合實驗一的結果寫的，要做實驗二請整段註解掉  
 
## training_experiment3.py

- based on day merge UUID event data  
