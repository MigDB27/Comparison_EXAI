#%%bibliotecas

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import gower
from aix360.algorithms.nncontrastive import NearestNeighborContrastiveExplainer

#%%import datasets

def juntar_df(X,y):
    df = pd.concat([X,y],axis=1)
    df.dropna(inplace=True)
    if len(df) > 2500:
        df=df.sample(n=2500,random_state=42)
    
    return df

#adutlt  
# fetch dataset 
adult = fetch_ucirepo(id=2) 
# data (as pandas dataframes) 
X_a = adult.data.features 
y_a = adult.data.targets 
df_a = juntar_df(X_a,y_a)

#statlog
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X_s = statlog_german_credit_data.data.features 
y_s = statlog_german_credit_data.data.targets 
df_s = juntar_df(X_s,y_s)

#default
# fetch dataset 
default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
# data (as pandas dataframes) 
X_d = default_of_credit_card_clients.data.features 
y_d = default_of_credit_card_clients.data.targets   
df_d = juntar_df(X_d,y_d)

#bank
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X_b = bank_marketing.data.features 
y_b = bank_marketing.data.targets 
df_b = juntar_df(X_b,y_b)


#%%TRATAMENTO DE DADOS

#USAR ONEHOTENCODER

#adult
df_a = df_a.replace("?",np.nan)
df_a = df_a.replace("<=50K.","<=50K")
df_a = df_a.replace(">50K.",">50K")

le_a1 = LabelEncoder()
le_a2 = LabelEncoder()
le_a3 = LabelEncoder()
le_a4 = LabelEncoder()
le_a5 = LabelEncoder()
le_a6 = LabelEncoder()
le_a7 = LabelEncoder()
le_a8 = LabelEncoder()
le_a9 = LabelEncoder()

df_a['workclass'] = le_a1.fit_transform(df_a['workclass']) #1 Male 0 Female
df_a['education'] = le_a2.fit_transform(df_a['education']) # 0<50k 1>50k
df_a['marital-status'] = le_a3.fit_transform(df_a['marital-status']) #1 Male 0 Female
df_a['occupation'] = le_a4.fit_transform(df_a['occupation']) # 0<50k 1>50k
df_a['relationship'] = le_a5.fit_transform(df_a['relationship']) #1 Male 0 Female
df_a['race'] = le_a6.fit_transform(df_a['race']) # 0<50k 1>50k
df_a['sex'] = le_a7.fit_transform(df_a['sex']) #1 Male 0 Female
df_a['income'] = le_a8.fit_transform(df_a['income']) # 0<50k 1>50k
df_a['native-country'] = le_a9.fit_transform(df_a['native-country']) # 0<50k 1>50k

'''
imputer = KNNImputer(n_neighbors=3)
df_a = pd.DataFrame(imputer.fit_transform(df_a),columns =df_a.columns)
'''
#statlog
le_s1 = LabelEncoder()
le_s2 = LabelEncoder()
le_s3 = LabelEncoder()
le_s4 = LabelEncoder()
le_s5 = LabelEncoder()
le_s6 = LabelEncoder()
le_s7 = LabelEncoder()
le_s8 = LabelEncoder()
le_s9 = LabelEncoder()
le_s10 = LabelEncoder()
le_s11 = LabelEncoder()
le_s12 = LabelEncoder()
le_s13 = LabelEncoder()

df_s['Attribute1'] = le_s1.fit_transform(df_s['Attribute1']) #1 Male 0 Female
df_s['Attribute3'] = le_s2.fit_transform(df_s['Attribute3']) #1 Male 0 Female
df_s['Attribute4'] = le_s3.fit_transform(df_s['Attribute4']) #1 Male 0 Female
df_s['Attribute6'] = le_s4.fit_transform(df_s['Attribute6']) #1 Male 0 Female
df_s['Attribute7'] = le_s5.fit_transform(df_s['Attribute7']) #1 Male 0 Female
df_s['Attribute9'] = le_s6.fit_transform(df_s['Attribute9']) #1 Male 0 Female
df_s['Attribute10'] = le_s7.fit_transform(df_s['Attribute10']) #1 Male 0 Female
df_s['Attribute12'] = le_s8.fit_transform(df_s['Attribute12']) #1 Male 0 Female
df_s['Attribute14'] = le_s9.fit_transform(df_s['Attribute14']) #1 Male 0 Female
df_s['Attribute15'] = le_s10.fit_transform(df_s['Attribute15']) #1 Male 0 Female
df_s['Attribute17'] = le_s11.fit_transform(df_s['Attribute17']) #1 Male 0 Female
df_s['Attribute19'] = le_s12.fit_transform(df_s['Attribute19']) #1 Male 0 Female
df_s['Attribute20'] = le_s13.fit_transform(df_s['Attribute20']) #1 Male 0 Female
df_s['class'].replace(2,0,inplace=True)

'''
df_s = pd.DataFrame(imputer.fit_transform(df_s),columns = df_s.columns)
'''
#default
#dados já transformados

#bank
le_b1 = LabelEncoder()
le_b2 = LabelEncoder()
le_b3 = LabelEncoder()
le_b4 = LabelEncoder()
le_b5 = LabelEncoder()
le_b6 = LabelEncoder()
le_b7 = LabelEncoder()
le_b8 = LabelEncoder()
le_b9 = LabelEncoder()
le_b10 = LabelEncoder()

df_b['job'] = le_b1.fit_transform(df_b['job']) #1 Male 0 Female
df_b['marital'] = le_b2.fit_transform(df_b['marital']) #1 Male 0 Female
df_b['education'] = le_b3.fit_transform(df_b['education']) #1 Male 0 Female
df_b['default'] = le_b4.fit_transform(df_b['default']) #1 Male 0 Female
df_b['housing'] = le_b5.fit_transform(df_b['housing']) #1 Male 0 Female
df_b['loan'] = le_b6.fit_transform(df_b['loan']) #1 Male 0 Female
df_b['contact'] = le_b7.fit_transform(df_b['contact']) #1 Male 0 Female
df_b['month'] = le_b8.fit_transform(df_b['month']) #1 Male 0 Female
df_b['poutcome'] = le_b9.fit_transform(df_b['poutcome']) #1 Male 0 Female
df_b['y'] = le_b10.fit_transform(df_b['y']) #1 Male 0 Female
'''
df_b = pd.DataFrame(imputer.fit_transform(df_b),columns =df_b.columns)
'''


#%%MODELO

def check_label(df,model):
    X_col = df.columns.values[0:-1]
    y_col = df.columns.values[-1]
    df['predict'] = np.nan
    for i in range(len(df)):
        if df[X_col][i:i+1].isna().values.any() == True:
           df['predict'][i] = 'nan' 
        else:    
            df['predict'][i] = 'ok'

    return None
        
def avaliacao(df1,df2,df3,continuous):
    global a,b,IV
    
    
    if len(df1)!=len(df2):
        raise ValueError('Dfs com tamanho diferente')
        
    ok = 0  
    erro=0
    
    scaler1 = StandardScaler()

    df1 = pd.DataFrame(scaler1.fit_transform(df1),columns = df1.columns)
    df2 = pd.DataFrame(scaler1.transform(df2),columns = df2.columns)
    
    #df1_g = df1.copy()
    #df2_g = df2.copy()
    
    '''
    #IMPORTANTE
    categorical = df1.columns.difference(continuous)
    for col in df1.columns:
        if col in categorical:
            df1[col] = df1[col].astype(str)
            df2[col] = df2[col].astype(str)   
    '''
    
    for i in range(len(df1)):
        hamming = 0   
        a = df1[i:i+1].values.flatten().astype('float32')
        b = df2[i:i+1].values.flatten().astype('float32')
        #a = df1[i]
        #b = df2[i]
        
        '''
        concat = np.concatenate((a.reshape(1,-1),b.reshape(1,-1)))
        gower_aux = pd.DataFrame(concat)
        #gower_array = gower.gower_matrix(gower_aux)
        gower_result = gower.gower_matrix(gower_aux)[0][1]
        df3['gower'][i] = gower_result
    '''
        try:
            df3['euclidean'][i] = distance.euclidean(a,b)   
        except:
            df3['euclidean'][i] = np.nan 
    
        if df3['predict'][i] == 'ok':
            ok+=1
        if df3['predict'][i] == 'erro':
            erro+=1      
        
        for j in range(df1.shape[1]):
            if abs(df1.iat[i,j] - df2.iat[i,j]) != 0:
                hamming+=1     
        '''        
        for j in range(df1.shape[1]):
            if df1[i,j] != df2[i,j]:
                hamming+=1'''
    
        df3['hamming'][i] = hamming
        
    df3['cobertura_pct'] = (ok/len(df3))*100
    df3['erro_pct'] = (erro/len(df3))*100
    df3['euclidean_medio'] = df3['euclidean'].mean()
    df3['hamming_medio'] = df3['hamming'].mean()

    return print('FIM')

def delta_media(delta):
    delta.loc['media']= 0

    n_col_mean = 0
    sum_col_mean = 0
    
    #MUDANÇA NO CÓDIGO
    for col in delta.columns:
        for i in range(len(delta)-1):
            if (delta[col][i] != 0) and (delta[col][i] != np.nan) :
                n_col_mean += 1
                delta[col]['media'] += delta[col][i]
        delta[col]['media'] = delta[col]['media']/n_col_mean
        n_col_mean = 0

    return None
    

def result (test,df,continuous):
    #test é o df x_test
    #df são os cfs
    global test_aux,df_aux
    
    test_aux = test.copy()
    df_aux = df[test_aux.columns]
    test = test.reset_index(drop=True)
    
    scaler_delta = StandardScaler()
    test_sc = pd.DataFrame(scaler_delta.fit_transform(test),columns = test.columns)
    df_aux_sc = pd.DataFrame(scaler_delta.transform(df_aux),columns = df_aux.columns)
    
    delta =  pd.DataFrame(np.nan, index = np.arange(len(test)), columns = test.columns)
    delta_o =  pd.DataFrame(np.nan, index = np.arange(len(test)), columns = test.columns)
    change =  pd.DataFrame(np.nan, index = np.arange(len(test)), columns = test.columns)

    for i in range(len(test)):
        
        for j in range(len(test.columns)):
            if abs(test.iat[i,j] - df_aux.iat[i,j]) != 0:
                if np.isnan(test.iat[i,j]) == False:
                    change.iat[i,j] = 1
                else:
                    change.iat[i,j] = 0


        for col in test.columns:
            if col in continuous:
                delta[col][i] = abs(test_sc[col][i] - df_aux_sc[col][i])
                delta_o[col][i] = abs(test[col][i] - df_aux[col][i])
            else:
                if test[col][i] != df[col][i]:
                    delta[col][i] = 1
                    delta_o[col][i] = 1
                else:
                    delta[col][i] = 0
                    delta_o[col][i] =0
    
    change.loc['total']= change.sum()
    delta_media(delta)
    delta_media(delta_o)

    
    #delta = pd.DataFrame(scaler_delta.fit_transform(delta),columns = delta.columns)
      
    return (delta,delta_o,change)    
    

def nnce_auto(df,continuous,outcome,ml_model):
        
    scaler_nnce = StandardScaler()
        
    
    y = df[outcome]
    X = df.drop(columns = outcome)
    
    X = pd.DataFrame(scaler_nnce.fit_transform(X),columns = X.columns)


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, stratify=y, random_state=0)
    
   
    ml_model.fit(X_train, y_train)
    # Save model into JSON format.
    y_pred = ml_model.predict(X_test)
   #y_pred_proba = xgboost.predict_proba(X_test)   
    
    print(classification_report(y_test, y_pred))
    
    
    """##EXAI
    
    ###Nearest Neighbor Contrastive Explainer - IB
    
    https://github.com/Trusted-AI/AIX360/blob/master/examples/README.md
    """  
    
    cfs = pd.DataFrame(columns =  X.columns)
    for i in range(len(X_test)):   
        print("\nRODANDO PASSO {}".format(i))
        
        try:
            ind = i
            sample_neg = X_test.iloc[ind:ind+1]
            
            epochs = 100
            embedding_dim = 4
            layers_config = []
            random_seed = 1 # to have consistent resutls for demos
            neighbors = 1
            explainer_with_model = NearestNeighborContrastiveExplainer(model=ml_model.predict,
                                                          embedding_dim=embedding_dim,
                                                          layers_config=layers_config,
                                                          neighbors=neighbors)
            
            
            history = explainer_with_model.fit(
                X_train,
                epochs=epochs,
                numeric_scaling=None,
                random_seed=random_seed,
            )
            
            p_str = f'Epochs: {epochs}'
            for key in history.history:
                p_str = f'{p_str}\t{key}: {history.history[key][-1]:.4f}'
            print(p_str)
            
            explainer_with_model._embedding._encoder.summary()
            
            # find nearest benign contrastive
            nearest_pos_contrastive = explainer_with_model.explain_instance(sample_neg)
            #nearest_pos_contrastive[0]["neighbors"][0]
            
            
            cfs.loc[len(cfs)] = nearest_pos_contrastive[0]["neighbors"][0]       
            
        except:
            cfs.loc[len(cfs)] = np.nan     
    

    cfs = pd.DataFrame(scaler_nnce.inverse_transform(cfs),columns = cfs.columns)
    cfs_target = y_test.reset_index(drop=True).copy()
    target = outcome
    for i in range(len(cfs_target)):
        if cfs_target[i] == 1:
            cfs_target[i] = 0
        else:
            cfs_target[i] = 1
    cfs2 = pd.concat([cfs,cfs_target],axis=1)
    

    #cfs2.to_csv('C:\\Users\\Miguel\\Documents\\Programacao\\py\\dissertacao\\resultados\\dataset_corrigido\\nnce_mushroom_df.csv')
    
    
    test_aux = pd.DataFrame(scaler_nnce.inverse_transform(X_test),columns=X_test.columns)
    #test_aux = pd.concat([test_aux,y_test],axis=1)
    #test_aux = test_aux.drop(columns = outcome)
    
    (delta,delta_o,change)= result(test_aux, cfs2, continuous)
    #test_aux = pd.concat([X_test,y_test],axis=1)
    #cfs2_sc = pd.DataFrame(scaler_nnce.inverse_transform(cfs2),columns = cfs2.columns)

    ##########################
    check_label(cfs2,ml_model)
    
    #########################################
    df_cfs = cfs2.copy()

    col = df_cfs.columns[0:-2]

    df_cfs['euclidean'] = np.nan
    df_cfs['euclidean_medio'] = np.nan
    df_cfs['hamming'] = np.nan
    df_cfs['hamming_medio'] = np.nan
    df_cfs['cobertura_pct'] = np.nan
    df_cfs['erro_pct'] = np.nan

    X_cfs=cfs2[col]
        
    #x_test2 = x_test.reset_index(drop=True)
    
    avaliacao(X_cfs,X_test,df_cfs,continuous)

    
    return (delta,delta_o,change,df_cfs)
    
    
    #CFS ESTÃO INCORRETOS!!! FAZER PREDICT!
#%%rodar

#adult
continuous_a = ['age','fnlwgt','capital-gain','capital-loss','hours-per-week']
outcome_a = 'income'

#statlog
continuous_s = ['Attribute2','Attribute5','Attribute8','Attribute11','Attribute13','Attribute16','Attribute18']
outcome_s = 'class'

#default
continuous_d = ['X1','X5','X6','X7','X8','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','X19','X20','X21','X22','X23']
outcome_d = 'Y'

#bank

continuous_b = ['age','balance','duration','campaign','pdays','previous']
outcome_b = 'y'
'''
X_s2 = df_s.drop(columns = outcome_s)
y_s2 = df_s[outcome_s]

X_a2 = df_a.drop(columns = outcome_a)
y_a2 = df_a[outcome_a]

X_d2 = df_d.drop(columns = outcome_d)
y_d2 = df_d[outcome_d]

X_b2 = df_b.drop(columns = outcome_b)
y_b2 = df_b[outcome_b]'''

 
#ml_model = xgb.XGBClassifier()
#fatiamento

#cfs_s = nnce_auto(df_s,xgb.XGBClassifier(),outcome_s,continuous_s)
#cfs_a = nnce_auto(df_a,xgb.XGBClassifier(),outcome_a,continuous_a)
#cfs_d = nnce_auto(df_d,xgb.XGBClassifier(),outcome_d,continuous_d)
#cfs_b = nnce_auto(df_b,xgb.XGBClassifier(),outcome_b,continuous_b)

(delta_s,delta_so,change_s,cfs_s) = nnce_auto (df_s,continuous_s,outcome_s,xgb.XGBClassifier())
#(delta_a,delta_ao,change_a,cfs_a) = nnce_auto (df_a,continuous_a,outcome_a,xgb.XGBClassifier())
#(delta_d,delta_do,change_d,cfs_d) = nnce_auto (df_d,continuous_d,outcome_d,xgb.XGBClassifier())
#(delta_b,delta_bo,change_b,cfs_b) = nnce_auto (df_b,continuous_b,outcome_b,xgb.XGBClassifier())

#(delta_sr,delta_sro,change_sr,cfs_sr) = nnce_auto (df_s,continuous_s,outcome_s,RandomForestClassifier())
#(delta_sa,delta_sao,change_sa,cfs_sa) = nnce_auto (df_s,continuous_s,outcome_s,AdaBoostClassifier())



'''
import matplotlib.pyplot as plt


plot_classification_contour(X_test, clf)
plt.scatter(X_train[:, 0], X_train[:, 1], marker='o', edgecolors='k', alpha=0.5, color=[['orange', 'blue'][i] for i in y_train])
plt.scatter(obs[:,0], obs[:,1], marker='o', color='lime', edgecolors='k', s=100)
plt.scatter(cf_x[:,0], cf_x[:,1], marker='o', color='purple', edgecolors='k', s=100)
plt.tight_layout()

CF.enemy

obs[0]'''

#%%

def nan_corrigir(df1,df2,df3):
    for i in range(len(df1)):
        if df1['predict'][i] == 'nan':
            df2.loc[i] = np.nan            
            df3.loc[i] = np.nan  
    return None

#nan_corrigir(cfs_a,delta_a,change_a)
nan_corrigir(cfs_s,delta_s,change_s)
#nan_corrigir(cfs_d,delta_d,change_d)
#nan_corrigir(cfs_b,delta_b,change_b)
#nan_corrigir(cfs_sr,delta_sr,change_sr)
#nan_corrigir(cfs_sa,delta_sa,change_sa)
#%%INVERSE

#cfs_a.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_adult.csv',sep=';',decimal=',')
cfs_s.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_statlog.csv',sep=';',decimal=',')
#cfs_d.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_default.csv',sep=';',decimal=',')
#cfs_b.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_bank.csv',sep=';',decimal=',')
#cfs_sr.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_statlog_rf.csv',sep=';',decimal=',')
#cfs_sa.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_statlog_ada.csv',sep=';',decimal=',')

#delta_a.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_nnce_adult.csv',sep=';',decimal=',')
delta_s.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_nnce_statlog.csv',sep=';',decimal=',')
#delta_d.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_nnce_default.csv',sep=';',decimal=',')
#delta_b.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_nnce_bank.csv',sep=';',decimal=',')
#delta_sr.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_nnce_statlog_rf.csv',sep=';',decimal=',')
#delta_sa.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_nnce_statlog_ada.csv',sep=';',decimal=',')

#delta_ao.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_o_nnce_adult.csv',sep=';',decimal=',')
delta_so.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_o_nnce_statlog.csv',sep=';',decimal=',')
#delta_do.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_o_nnce_default.csv',sep=';',decimal=',')
#delta_bo.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_o_nnce_bank.csv',sep=';',decimal=',')
#delta_sro.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_o_nnce_statlog_rf.csv',sep=';',decimal=',')
#delta_sao.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\delta_o_nnce_statlog_ada.csv',sep=';',decimal=',')

#change_a.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\change_nnce_adult.csv',sep=';',decimal=',')
change_s.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\change_nnce_statlog.csv',sep=';',decimal=',')
#change_d.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\change_nnce_default.csv',sep=';',decimal=',')
#change_b.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\change_nnce_bank.csv',sep=';',decimal=',')
#change_sr.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\change_nnce_statlog_rf.csv',sep=';',decimal=',')
#change_sa.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\change_nnce_statlog_ada.csv',sep=';',decimal=',')



#%%

#cfs_a.dropna(inplace=True)
#cfs_s.dropna(inplace=True)
#cfs_d.dropna(inplace=True)
#cfs_b.dropna(inplace=True)
#cfs_sa.dropna(inplace=True)
#cfs_sr.dropna(inplace=True)


'''
cfs_a['workclass'] = le_a1.inverse_transform(cfs_a['workclass'].astype(int)) #1 Male 0 Female
cfs_a['education'] = le_a2.inverse_transform(cfs_a['education'].astype(int)) # 0<50k 1>50k
cfs_a['marital-status'] = le_a3.inverse_transform(cfs_a['marital-status'].astype(int)) #1 Male 0 Female
cfs_a['occupation'] = le_a4.inverse_transform(cfs_a['occupation'].astype(int)) # 0<50k 1>50k
cfs_a['relationship'] = le_a5.inverse_transform(cfs_a['relationship'].astype(int)) #1 Male 0 Female
cfs_a['race'] = le_a6.inverse_transform(cfs_a['race'].astype(int)) # 0<50k 1>50k
cfs_a['sex'] = le_a7.inverse_transform(cfs_a['sex'].astype(int)) #1 Male 0 Female
cfs_a['income'] = le_a8.inverse_transform(cfs_a['income'].astype(int)) # 0<50k 1>50k
cfs_a['native-country'] = le_a9.inverse_transform(cfs_a['native-country'].astype(int)) # 0<50k 1>50k

cfs_s['Attribute1'] = le_s1.inverse_transform(cfs_s['Attribute1'].astype(int)) #1 Male 0 Female
cfs_s['Attribute3'] = le_s2.inverse_transform(cfs_s['Attribute3'].astype(int)) #1 Male 0 Female
cfs_s['Attribute4'] = le_s3.inverse_transform(cfs_s['Attribute4'].astype(int)) #1 Male 0 Female
cfs_s['Attribute6'] = le_s4.inverse_transform(cfs_s['Attribute6'].astype(int)) #1 Male 0 Female
cfs_s['Attribute7'] = le_s5.inverse_transform(cfs_s['Attribute7'].astype(int)) #1 Male 0 Female
cfs_s['Attribute9'] = le_s6.inverse_transform(cfs_s['Attribute9'].astype(int)) #1 Male 0 Female
cfs_s['Attribute10'] = le_s7.inverse_transform(cfs_s['Attribute10'].astype(int)) #1 Male 0 Female
cfs_s['Attribute12'] = le_s8.inverse_transform(cfs_s['Attribute12'].astype(int)) #1 Male 0 Female
cfs_s['Attribute14'] = le_s9.inverse_transform(cfs_s['Attribute14'].astype(int)) #1 Male 0 Female
cfs_s['Attribute15'] = le_s10.inverse_transform(cfs_s['Attribute15'].astype(int)) #1 Male 0 Female
cfs_s['Attribute17'] = le_s11.inverse_transform(cfs_s['Attribute17'].astype(int)) #1 Male 0 Female
cfs_s['Attribute19'] = le_s12.inverse_transform(cfs_s['Attribute19'].astype(int)) #1 Male 0 Female
cfs_s['Attribute20'] = le_s13.inverse_transform(cfs_s['Attribute20'].astype(int)) #1 Male 0 Female

cfs_sr['Attribute1'] = le_s1.inverse_transform(cfs_sr['Attribute1'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute3'] = le_s2.inverse_transform(cfs_sr['Attribute3'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute4'] = le_s3.inverse_transform(cfs_sr['Attribute4'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute6'] = le_s4.inverse_transform(cfs_sr['Attribute6'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute7'] = le_s5.inverse_transform(cfs_sr['Attribute7'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute9'] = le_s6.inverse_transform(cfs_sr['Attribute9'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute10'] = le_s7.inverse_transform(cfs_sr['Attribute10'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute12'] = le_s8.inverse_transform(cfs_sr['Attribute12'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute14'] = le_s9.inverse_transform(cfs_sr['Attribute14'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute15'] = le_s10.inverse_transform(cfs_sr['Attribute15'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute17'] = le_s11.inverse_transform(cfs_sr['Attribute17'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute19'] = le_s12.inverse_transform(cfs_sr['Attribute19'].astype(int)) #1 Male 0 Female
cfs_sr['Attribute20'] = le_s13.inverse_transform(cfs_sr['Attribute20'].astype(int)) #1 Male 0 Female

cfs_sa['Attribute1'] = le_s1.inverse_transform(cfs_sa['Attribute1'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute3'] = le_s2.inverse_transform(cfs_sa['Attribute3'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute4'] = le_s3.inverse_transform(cfs_sa['Attribute4'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute6'] = le_s4.inverse_transform(cfs_sa['Attribute6'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute7'] = le_s5.inverse_transform(cfs_sa['Attribute7'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute9'] = le_s6.inverse_transform(cfs_sa['Attribute9'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute10'] = le_s7.inverse_transform(cfs_sa['Attribute10'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute12'] = le_s8.inverse_transform(cfs_sa['Attribute12'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute14'] = le_s9.inverse_transform(cfs_sa['Attribute14'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute15'] = le_s10.inverse_transform(cfs_sa['Attribute15'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute17'] = le_s11.inverse_transform(cfs_sa['Attribute17'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute19'] = le_s12.inverse_transform(cfs_sa['Attribute19'].astype(int)) #1 Male 0 Female
cfs_sa['Attribute20'] = le_s13.inverse_transform(cfs_sa['Attribute20'].astype(int)) #1 Male 0 Female

cfs_b['job'] = le_b1.inverse_transform(cfs_b['job'].astype(int)) #1 Male 0 Female
cfs_b['marital'] = le_b2.inverse_transform(cfs_b['marital'].astype(int)) #1 Male 0 Female
cfs_b['education'] = le_b3.inverse_transform(cfs_b['education'].astype(int)) #1 Male 0 Female
cfs_b['default'] = le_b4.inverse_transform(cfs_b['default'].astype(int)) #1 Male 0 Female
cfs_b['housing'] = le_b5.inverse_transform(cfs_b['housing'].astype(int)) #1 Male 0 Female
cfs_b['loan'] = le_b6.inverse_transform(cfs_b['loan'].astype(int)) #1 Male 0 Female
cfs_b['contact'] = le_b7.inverse_transform(cfs_b['contact'].astype(int)) #1 Male 0 Female
cfs_b['month'] = le_b8.inverse_transform(cfs_b['month'].astype(int)) #1 Male 0 Female
cfs_b['poutcome'] = le_b9.inverse_transform(cfs_b['poutcome'].astype(int)) #1 Male 0 Female
cfs_b['y'] = le_b10.inverse_transform(cfs_b['y'].astype(int)) #1 Male 0 Female

''''''
#%% teste
cfs = pd.DataFrame(columns=X.columns)


#idx = np.random.randint(X_test.shape[0])
ind = 1
#obs = X_test[idx, :].reshape(1, -1)
obs = X_test.iloc[ind,:].to_numpy().reshape(1,-1)  
#CF = cf.CounterfactualExplanation(obs, clf.predict, method='GS')
CF = CounterfactualExplanation(obs, xgboost.predict, method='GS')
CF.fit(n_in_layer=200, first_radius=1.1, dicrease_radius=2.0, sparse=False, verbose=False)
cf_x = CF.enemy.reshape(1, -1)[0]
cfs.loc[len(cfs)] = cf_x'''





#%%teste

#cfs_a.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_adult_na.csv',sep=';',decimal=',')
#cfs_s.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_statlog_na.csv',sep=';',decimal=',')
#cfs_d.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_default_na.csv',sep=';',decimal=',')
#cfs_b.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_bank_na.csv',sep=';',decimal=',')
#cfs_sr.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_statlog_rf_na.csv',sep=';',decimal=',')
#cfs_sa.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\NNCE\\cf_nnce_statlog_ada_na.csv',sep=';',decimal=',')

print('FIM DO PROCESSO')