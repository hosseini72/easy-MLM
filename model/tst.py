import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.sparse import load_npz


addrss=r'O:\Second Semister\dissertation\dis-dataset\liar_dataset\train_data\Liar_bow.npz'
label_address=r'O:\Second Semister\dissertation\dis-dataset\liar_dataset\train_data\label.npz'
label= pd.read_csv(r'O:\Second Semister\dissertation\dis-dataset\liar_dataset\train_data\Liar_label.csv')
# label= liar['label']
data= load_npz(addrss)
# label= scipy.sparse.load_npz(r'O:\Second Semister\dissertation\dis-dataset\liar_dataset\train_data\label.npz')
split=None
# label= np.load(label_address)
if split is None:
    train_data, test_data, train_labels, test_labels = train_test_split(data, label, test_size= 0.2, random_state=42) 


# print(split[0].shape)
# print(split[2])

# print(split[1].shape)
# print(split[3].shape)

print(test_labels.shape)