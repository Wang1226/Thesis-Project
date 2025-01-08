
import pandas as pd
from sklearn import preprocessing as pp


columns_name=['user_id','item_id','rating','timestamp']
# df = pd.read_csv("./ml-100k/u.data",sep='\t',names=columns_name)
df=pd.read_csv('./ml-latest-small/ratings.csv')
df.columns=columns_name

le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()
df.loc[:, 'user_id_2'] = le_user.fit_transform(df['user_id'].values)
df.loc[:, 'item_id_2'] = le_item.fit_transform(df['item_id'].values)

item_sim=pd.read_csv('dataset_movie/cosine_similarity_after_fit_transform.csv',index_col=0)
item_sim.index = item_sim.index.astype(int)
item_sim.columns = item_sim.columns.astype(int)

intersection_sim_matrix=pd.read_csv('./dataset_movie/Jaccard_sim_matrix.csv',index_col=0)

