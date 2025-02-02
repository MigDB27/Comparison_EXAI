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

import dice_ml
from dice_ml.utils import helpers  # helper functions

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import gower

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


#%%DICE

#NÃO FOI APLICADO ONEHOTENCODER

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
        df3['euclidean'][i] = distance.euclidean(a,b)   
    
    
        if df3['predict'][i] == 'ok':
            ok+=1
        if df3['predict'][i] == 'erro':
            erro+=1      
        
        for j in range(df1.shape[1]):
            if df1.iat[i,j] != df2.iat[i,j]:
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
    
    for col in delta.columns:
        for i in range(len(delta)-1):
            if delta[col][i] != 0:
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
            if test.iat[i,j] != df_aux.iat[i,j]:
                change.iat[i,j] = 1

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


def dice_auto (df,X,continuous,outcome,ml_model):
    #df é o dataframe
    #continuous lista com os nomes dos atributos continuos 
    #nome da coluna alvo
    #modelo de IA ex: xgb.XGBClassifier()
    
    #df = pd.concat([df_train,df_test])
    d = dice_ml.Data(dataframe=df, continuous_features = continuous, outcome_name=outcome)
    
    df.reset_index(drop=True,inplace=True)
    df
    
    target = df[outcome]
    # Split data into train and test
    datasetX = df.drop(outcome, axis=1)
    x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=target)
    
    #IMPORTANTE
    imputer = KNNImputer(n_neighbors=3)
    x_train = pd.DataFrame(imputer.fit_transform(x_train),columns =x_train.columns)
    x_test.dropna(inplace=True)
    
    numerical = x_train.columns
    categorical = x_train.columns.difference(numerical)
    
    
    # We create the preprocessing pipelines for both numeric and categorical data.
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    transformations = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical),
            ('cat', categorical_transformer, categorical)
            ])
    
    # Append classifier to preprocessing pipeline.
    # Now we have a full prediction pipeline.
    
    clf = Pipeline(steps=[('preprocessor', transformations),
                          ('classifier',ml_model)])
    model = clf.fit(x_train, y_train)
    
    # provide the trained ML model to DiCE's model object
    backend = 'sklearn'
    m = dice_ml.Model(model=model, backend=backend)
    
    # initiate DiCE
    exp_random = dice_ml.Dice(d, m, method="random")
    
    cfs = pd.DataFrame(columns = X.columns)
    
    ######################
       
    for i in range(len(x_test)):
        
        print("\nRODANDO PASSO {}".format(i))
        try:    
            query_instances = x_test[i:i+1]
            # generate counterfactuals
            dice_exp_random = exp_random.generate_counterfactuals(query_instances, total_CFs=1, desired_class="opposite", verbose=False)
            dice_exp_random.visualize_as_dataframe(show_only_changes=True)
            dice_exp_random.visualize_as_list()
            cf = dice_exp_random.cf_examples_list[0].final_cfs_df
            cfs = pd.concat([cfs,cf],axis=0)
        except:
            cfs.loc[len(cfs)] = np.nan
    
    cfs.reset_index(drop=True,inplace=True)
    #ver na doc features_to_vary
    #####
    
    (delta,delta_o,change)= result(x_test, cfs, continuous)
    
    ##########################
    check_label(cfs,model)
    
    #########################################
    df_cfs = cfs.copy()

    col = df_cfs.columns[0:-2]

    df_cfs['euclidean'] = np.nan
    df_cfs['euclidean_medio'] = np.nan
    df_cfs['hamming'] = np.nan
    df_cfs['hamming_medio'] = np.nan
    df_cfs['cobertura_pct'] = np.nan
    df_cfs['erro_pct'] = np.nan

    X_cfs=df_cfs[col]
    
    #x_test2 = x_test.reset_index(drop=True)
    
    avaliacao(X_cfs,x_test,df_cfs,continuous)

    #df_cfs.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\dice_' + dataset + '.csv')
    
    return (delta,delta_o,change,df_cfs)

#%%rodar

#ml_model = xgb.XGBClassifier()

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
cfs_a = dice_auto (df_a,X_a,continuous_a,outcome_a,xgb.XGBClassifier())
cfs_s = dice_auto (df_s,X_s,continuous_s,outcome_s,xgb.XGBClassifier())
cfs_d = dice_auto (df_d,X_d,continuous_d,outcome_d,xgb.XGBClassifier())
cfs_b = dice_auto (df_b,X_b,continuous_b,outcome_b,xgb.XGBClassifier())
'''
(delta_s,delta_so,change_s,cfs_s) = dice_auto (df_s,X_s,continuous_s,outcome_s,xgb.XGBClassifier())
#(delta_a,delta_ao,change_a,cfs_a) = dice_auto (df_a,X_a,continuous_a,outcome_a,xgb.XGBClassifier())
#(delta_d,delta_do,change_d,cfs_d) = dice_auto (df_d,X_d,continuous_d,outcome_d,xgb.XGBClassifier())
#(delta_b,delta_bo,change_b,cfs_b) = dice_auto (df_b,X_b,continuous_b,outcome_b,xgb.XGBClassifier())


#(delta_sr,delta_sro,change_sr,cfs_sr) = dice_auto (df_s,X_s,continuous_s,outcome_s,RandomForestClassifier())
#(delta_sa,delta_sao,change_sa,cfs_sa) = dice_auto (df_s,X_s,continuous_s,outcome_s,AdaBoostClassifier())



#%%INVERSE
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
'''
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
'''
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
'''
#%%
#cfs_a.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\cf_dice_adult.csv',sep=';',decimal=',')
cfs_s.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\cf_dice_statlog.csv',sep=';',decimal=',')
#cfs_d.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\cf_dice_default.csv',sep=';',decimal=',')
#cfs_b.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\cf_dice_bank.csv',sep=';',decimal=',')
#cfs_sr.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\cf_dice_statlog_rf.csv',sep=';',decimal=',')
#cfs_sa.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\cf_dice_statlog_ada.csv',sep=';',decimal=',')

#delta_a.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_dice_adult.csv',sep=';',decimal=',')
delta_s.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_dice_statlog.csv',sep=';',decimal=',')
#delta_d.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_dice_default.csv',sep=';',decimal=',')
#delta_b.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_dice_bank.csv',sep=';',decimal=',')
#delta_sr.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_dice_statlog_rf.csv',sep=';',decimal=',')
#delta_sa.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_dice_statlog_ada.csv',sep=';',decimal=',')

#delta_ao.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_o_dice_adult.csv',sep=';',decimal=',')
delta_so.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_o_dice_statlog.csv',sep=';',decimal=',')
#delta_do.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_o_dice_default.csv',sep=';',decimal=',')
#delta_bo.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_o_dice_bank.csv',sep=';',decimal=',')
#delta_sro.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_o_dice_statlog_rf.csv',sep=';',decimal=',')
#delta_sao.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\delta_o_dice_statlog_ada.csv',sep=';',decimal=',')


#change_a.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\change_dice_adult.csv',sep=';',decimal=',')
change_s.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\change_dice_statlog.csv',sep=';',decimal=',')
#change_d.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\change_dice_default.csv',sep=';',decimal=',')
#change_b.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\change_dice_bank.csv',sep=';',decimal=',')
#change_sr.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\change_dice_statlog_rf.csv',sep=';',decimal=',')
#change_sa.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\DICE\\change_dice_statlog_ada.csv',sep=';',decimal=',')

print('FIM DO PROCESSO')