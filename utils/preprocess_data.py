import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import scipy.io
import os
   

def preprocess(data_path,data_name,social_neighb=True):
  #depending on if the dataset has social network graph or not
  #if edges features are vectors (text features) or float(ratings)
  u_list, i_list, social_list, ts_list, label_list = [], [], [], [], []
  feat_l = []
  idx_list = []
  trust_df = None
  users_in_social = []

  ext = 'mat' if data_name == "epinion" or "ciao" else 'csv'

  #columns positions
  if social_neighb == False: #TGN datasets
    pos_u, pos_i, pos_ts, pos_label , pos_feat = 0, 1, 2, 3, 4


  else: #ex: ml_trust_epinion
    
    trust_file_mat = data_path+'{}_trust.{}'.format(data_name,ext)
    trust_file_csv = data_path+'{}_trust.csv'.format(data_name)
    df_file_csv = data_path+'{}.{}'.format(data_name,"csv")
    df_file_mat = data_path+'{}.{}'.format(data_name,"mat")

    if not os.path.exists(df_file_csv): 
      data = scipy.io.loadmat(df_file_mat)
      for i in data:
        if '__' not in i and 'readme' not in i:
              np.savetxt((df_file_csv),data[i],delimiter=',')

    if not os.path.exists(trust_file_csv):
      data = scipy.io.loadmat(trust_file_mat)
      for i in data:
        if '__' not in i and 'readme' not in i:
              np.savetxt((trust_file_csv),data[i],delimiter=',')

    pos_u, pos_i, pos_ts, pos_label = 0, 1, 5, 3

    trust_f = np.loadtxt(trust_file_csv,dtype=float,delimiter=',')
    trust_df = pd.DataFrame(trust_f,columns=['u1','u2','tu'])

    users_in_social = set(trust_df.u1.values) | set(trust_df.u2.values)

  #features 
  with open(data_path+data_name+'.csv') as f:
    s = next(f)

    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(float(e[pos_u]))
      i = int(float(e[pos_i]))

      ts = float(e[pos_ts])
      label = float(e[pos_label])  # int(e[3])

      if social_neighb:
        if u in users_in_social: #filter users by keeping only thoses who are in social graph
          u_list.append(u)
          i_list.append(i)
          ts_list.append(ts)
          label_list.append(label)
          idx_list.append(idx)
        else:
          continue
      else:
        u_list.append(u)
        i_list.append(i)
        ts_list.append(ts)
        label_list.append(label)
        idx_list.append(idx)
        feat = np.array([float(x) for x in e[pos_feat:]])
        feat_l.append(feat)

    if social_neighb: #edges_features : ratings for GraphRec to transform into embeddings later; text features for TGN
      feat_l = label_list
      trust_df = trust_df[trust_df.u1.isin(u_list) & trust_df.u2.isin(u_list)]
      trust_df['e_idx'] = list(np.arange(trust_df.shape[0]))

  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l), trust_df



def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True, social_graph=True):
  print(os.getcwd())
  Path("data/").mkdir(parents=True, exist_ok=True)
  EXT = 'mat' if data_name == 'epinion' or "ciao" else 'csv'
  PATH = './data/'
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)
  TRUST_DF = './data/ml_{}_trust.csv'.format(data_name)


  df, feat , trust_df = preprocess(PATH,data_name,True)
  new_df = reindex(df, bipartite)

  dict_index = dict(zip(set(df.u.values), set(new_df.u.values)))

  trust_df.u1 = trust_df.u1.map(dict_index).values
  trust_df.u2 = trust_df.u2.map(dict_index).values

  print(trust_df.head())
  print(feat.shape)
  empty = np.zeros(feat.shape[-1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, 172))

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)

  if social_graph:
    trust_df.to_csv(TRUST_DF)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='epinion')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

parser.add_argument('--social_graph', type=bool, help='is social graph available',
                    default='True')
args = parser.parse_args()

run(args.data, bipartite=args.bipartite, social_graph=args.social_graph)