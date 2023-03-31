import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import *
sys.path.insert(0, 'C:/Users/hp/Downloads/MTP/web_app/Criteo')

from arnn_config import config1
import pickle

#load it
with open(f'config_class_arnn_criteo.pickle', 'rb') as file2:
    Config = pickle.load(file2)


# def data_preprocess_criteo(df, max_input=Config.embed_feature_number + Config.non_embed_feature_number, seq_max_len=Config.seq_max_len):
    
#     key_count = df.groupby(by=['key_ID','label'],sort=False).size().reset_index()
#     key_count.columns = ['key_ID','label','seq_len']
#     key_count_list = key_count.values

#     cat_cols1 = list(np.where(df.columns.str.find('cat')==0))
#     camp_col1 = list(np.where(df.columns.str.find('campaign')==0))              ## Campaign List column position
#     click_col1 = list(np.where(df.columns.str.find('click')==0))                ## Click List column  position
#     time_col1 = list(np.where(df.columns.str.find('time')==0))                  ## Click List column  position
#     temp = np.append(time_col1[0],click_col1[0])
#     temp = np.append(temp,camp_col1[0])
#     time_click_camp_cat_col = np.append(temp,cat_cols1[0])

#     df_values = df.iloc[:,time_click_camp_cat_col].values.tolist()

#     total_data   = []
#     total_seqlen = []
#     total_label  = []
#     total_keyID  = []
#     itr = 0
#     for keyID_label_seqlen in key_count_list:
#         tmpseq = []
#         try:
#             keyID, label, seq_len = keyID_label_seqlen
#             tmpseq = df_values[itr: itr+seq_len]
#             itr += seq_len
#             tmpseq = tmpseq[-1 * seq_max_len:]
#             # padding zero
#             if seq_len < seq_max_len:
#                 for _ in range(seq_len, seq_max_len):
#                     tmpseq.append([0] * max_input)         #right padding
#             if seq_len > seq_max_len:
#                 seq_len = seq_max_len
#         except Exception as e: 
#             print(e)
#             continue
#         total_keyID.append(keyID)
#         total_data.append(tmpseq)
#         total_seqlen.append(seq_len)
#         total_label.append(int(label))
#     x1,x2 = np.split(np.array(total_data), [Config.non_embed_feature_number,], 2) 
#     x2  = x2.astype('int')
#     return [x1, x2, np.array(total_seqlen,dtype ='int32').reshape(-1,1) ], np.array(total_label,dtype ='float32').reshape(-1,1),  np.array(total_keyID).reshape(-1,1)     

def data_preprocess_criteo(df, max_input=Config.embed_feature_number + Config.non_embed_feature_number, seq_max_len=Config.seq_max_len):
    
    key_count = df.groupby(by=['key_ID','label'],sort=False).size().reset_index()
    key_count.columns = ['key_ID','label','seq_len']
    key_count_list = key_count.values

    cat_cols1 = list(np.where(df.columns.str.find('cat')==0))
    camp_col1 = list(np.where(df.columns.str.find('campaign')==0))              ## Campaign List column position
    click_col1 = list(np.where(df.columns.str.find('click')==0))                ## Click List column  position
    time_col1 = list(np.where(df.columns.str.find('time')==0))                  ## Click List column  position
    cost_col1 = list(np.where(df.columns.str.find('cost')==0))                  ## Click List column  position
    temp = np.append(time_col1[0],click_col1[0])
    temp = np.append(temp,camp_col1[0])
    time_click_camp_cat_col = np.append(temp,cat_cols1[0])
    

    df_values = df.iloc[:,time_click_camp_cat_col].values.tolist()
    df_cost   = df.iloc[:,cost_col1[0]].values.tolist()
    
    total_data   = []
    total_seqlen = []
    total_label  = []
    total_keyID  = []
    total_cost   = []
    itr = 0
    for keyID_label_seqlen in key_count_list:
        tmpseq = []
        tmcost = []
        try:
            keyID, label, seq_len = keyID_label_seqlen
            tmpseq = df_values[itr: itr+seq_len]
            tmcost = df_cost[itr: itr+seq_len]
        
            itr += seq_len
            tmpseq = tmpseq[-1 * seq_max_len:]
            tmcost = tmcost[-1 * seq_max_len:]
            # padding zero
            if seq_len < seq_max_len:
                for _ in range(seq_len, seq_max_len):
                    tmpseq.append([0] * max_input)         #right padding
                    tmcost.append([0])
            if seq_len > seq_max_len:
                seq_len = seq_max_len
        except Exception as e: 
            print(e)
            continue
        total_keyID.append(keyID)
        total_data.append(tmpseq)
        total_seqlen.append(seq_len)
        total_label.append(int(label))
        total_cost.append(tmcost)
    x1,x2 = np.split(np.array(total_data), [Config.non_embed_feature_number,], 2) 
    x2  = x2.astype('int')
    return [x1, x2, np.array(total_seqlen,dtype ='int32').reshape(-1,1) ], np.array(total_label,dtype ='float32').reshape(-1,1),  np.array(total_keyID).reshape(-1,1), np.array(total_cost)*100000   


# ---------------------------------------------------- to get attr score ---------------------------------------------------------------
# def attribution_criteo(X, y, modell):
#     Channel_value = {}
#     Channel_time = {}
#     try:
#         x1, x2, seq_len = X
#         attention =  modell.predict(X, batch_size=Config.batch_size, verbose=0)
#         y = y.reshape(-1)
#         seq_len = seq_len.reshape(-1)
#         for i in range(len(attention)):       
#             if y[i] != 0:
#                 for j in range(seq_len[i]):
#                     index = x2[i,j,0]
#                     v = attention[i,j,0]                                     # [user,timestep,0]
#                     if index in Channel_value :
#                         Channel_value[index] += v
#                         Channel_time[index] += 1
#                     else:
#                         Channel_value[index] = v
#                         Channel_time[index] = 1
#     except Exception as e:
#         print(e)
#     temp_l = []
#     for key in Channel_value:
#         temp_l.append([key,str(Channel_value[key] / Channel_time[key]), str(Channel_time[key])])
#     dff = pd.DataFrame(temp_l, columns=['campaigner','mean_weight','frequency'])
#     return dff

def attribution_criteo(X, y, cost, modell):
    Channel_value = {}
    Channel_time = {}
    Channel_cost = {}
    try:
        x1, x2, seq_len = X
        attention =  modell.predict(X, batch_size=Config.batch_size, verbose=0)
        y = y.reshape(-1)
        seq_len = seq_len.reshape(-1)
        for i in range(len(attention)):       
            if y[i] != 0:
                for j in range(seq_len[i]):
                    index = x2[i,j,0]
                    v = attention[i,j,0]                                     # [user,timestep,0]
                    if index in Channel_value :
                        Channel_value[index] += v
                        Channel_time[index] += 1
                        Channel_cost[index] += cost[i,j]
                    else:
                        Channel_value[index] = v
                        Channel_time[index] = 1
                        Channel_cost[index] = cost[i,j]
    except Exception as e:
        print(e)
    temp_l = []
    for key in Channel_value:
        temp_l.append(["C_"+str(key), round(Channel_value[key] / Channel_time[key], 1), str(Channel_time[key]), round(Channel_cost[key][0] / Channel_time[key], 2)])
    dff = pd.DataFrame(temp_l, columns=['Channel','Score','frequency', 'Cost'])
    return dff