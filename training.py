import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression


 #To perform data splitting on the imported DATA
def split_data(data,ratio):
    shuffled=np.random.permutation(len(data))
    test_data_set=int(len(data)*ratio)
    test_indices=shuffled[:test_data_set]
    train_indices=shuffled[test_data_set:]
    return data.iloc[test_indices],data.iloc[train_indices]

if __name__ == '__main__':
    #To read the DATA From CSV file 
    df = pd.read_csv('data.csv')
    train,test = split_data(df,0.2)
    X_train=train[['fever','bodyPain','age','runnyNose','difficultyBreath']].to_numpy()
    X_test=test[['fever','bodyPain','age','runnyNose','difficultyBreath']].to_numpy()
    Y_train=train['infectionProb'].to_numpy().reshape(199,)
    Y_test=test['infectionProb'].to_numpy().reshape(800,)
    clf = LogisticRegression()
    clf.fit(X_train,Y_train)
    outfile = open('model.pk1','wb')
    pickle.dump(clf,outfile)
    outfile.close()
    