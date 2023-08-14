import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.sparse import load_npz
from scipy import sparse
import os

# addrss=r'O:\Second Semister\dissertation\dis-dataset\liar_dataset\train_data\Liar_bow.npz'
# label_address=r'O:\Second Semister\dissertation\dis-dataset\liar_dataset\train_data\label.npz'
# label= pd.read_csv(r'O:\Second Semister\dissertation\dis-dataset\liar_dataset\train_data\Liar_label.csv')
# # label= liar['label']
# data= load_npz(addrss)
# # label= scipy.sparse.load_npz(r'O:\Second Semister\dissertation\dis-dataset\liar_dataset\train_data\label.npz')
# split=None
# # label= np.load(label_address)
# if split is None:
#     train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size= 0.2, random_state=42) 


# print(split[0].shape)
# print(split[2])

# print(split[1].shape)
# print(split[3].shape)

# print(type(test_labels.values))
# print(test_labels.values.ravel())


# def extention(adrs):
#     result= adrs.rsplit('.')
#     return result



# data_addr=r'O:\Second Semister\dissertation\dis-dataset\GossioCop\train_data'


# dt_adrs= os.path.join(data_addr, 'gossipcop_bow.npz')
# # lbl_adrs= os.path.join(data_addr, label_dt)
# x= extention(dt_adrs)[-1]
# print(x)


X_train= load_npz(r'O:\Second Semister\dissertation\dis-dataset\GossioCop\train_data\Liar_enhanc_bow_w2v.npz')
print(type(X_train))
if isinstance(X_train, sparse._csr.csr_matrix):
    X_train= X_train.toarray()
    print(type(X_train))


