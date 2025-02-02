#%%bibliotecas

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
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


#%%GSG

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
   


def gsg_auto(X,y,continuous,outcome,ml_model):
       
    scaler_gsg = StandardScaler()
    X = pd.DataFrame(scaler_gsg.fit_transform(X),columns = X.columns)
       
    from sklearn.model_selection import train_test_split
    import xgboost as xgb
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, stratify=y, random_state=0)
    
    """##Modelos ML
    
    ###XGBoost
    """
    
    # Use "hist" for constructing the trees, with early stopping enabled.
    #xgboost = xgb.XGBClassifier()
    # Fit the model, test sets are used for early stopping.
    ml_model.fit(X_train, y_train)
    # Save model into JSON format.
    y_pred = ml_model.predict(X_test)
    y_pred_proba = ml_model.predict_proba(X_test)
    
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))
    
    """###RF"""
    '''
    from sklearn.ensemble import RandomForestClassifier
    
    rf = RandomForestClassifier(max_depth=2, random_state=0)
    rf.fit(X_train, y_train)
    RandomForestClassifier(...)
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    
    print(classification_report(y_test, y_pred))'''
    
    """##EXAI
    
    """###Growing Spheres -HSS"""
    
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    
    import numpy as np
    from scipy.stats import kendalltau
    from sklearn.metrics.pairwise import pairwise_distances
    import matplotlib.pyplot as plt
    
    def get_distances(x1, x2, metrics=None):
        x1, x2 = x1.reshape(1, -1), x2.reshape(1, -1)
        euclidean = pairwise_distances(x1, x2)[0][0]
        same_coordinates = sum((x1 == x2)[0])
    
        #pearson = pearsonr(x1, x2)[0]
        kendall = kendalltau(x1, x2)
        out_dict = {'euclidean': euclidean,
                    'sparsity': x1.shape[1] - same_coordinates,
                    'kendall': kendall
                   }
        return out_dict
    
    
    def generate_ball(center, r, n):
        def norm(v):
            return np.linalg.norm(v, ord=2, axis=1)
        d = center.shape[1]
        u = np.random.normal(0,1,(n, d+2))  # an array of (d+2) normally distributed random variables
        norm_ = norm(u)
        u = 1/norm_[:,None]* u
        x = u[:, 0:d] * r #take the first d coordinates
        x = x + center
        return x
    
    def generate_ring(center, segment, n):
        def norm(v):
            return np.linalg.norm(v, ord=2, axis=1)
        d = center.shape[1]
        z = np.random.normal(0, 1, (n, d))
        try:
            u = np.random.uniform(segment[0]**d, segment[1]**d, n)
        except OverflowError:
            raise OverflowError("Dimension too big for hyperball sampling. Please use layer_shape='sphere' instead.")
        r = u**(1/float(d))
        z = np.array([a * b / c for a, b, c in zip(z, r,  norm(z))])
        z = z + center
        return z
    
    def generate_sphere(center, r, n):
        def norm(v):
                return np.linalg.norm(v, ord=2, axis=1)
        d = center.shape[1]
        z = np.random.normal(0, 1, (n, d))
        z = z/(norm(z)[:, None]) * r + center
        return z
    
    #!/usr/bin/env python
    # -*- coding: utf-8 -*-
    
    #from .utils.gs_utils import generate_ball, generate_sphere, generate_ring, get_distances
    from itertools import combinations
    import numpy as np
    from sklearn.metrics.pairwise import pairwise_distances
    from sklearn.utils import check_random_state
    
    
    
    class GrowingSpheres:
        """
        class to fit the Original Growing Spheres algorithm
    
        Inputs:
        obs_to_interprete: instance whose prediction is to be interpreded
        prediction_fn: prediction function, must return an integer label
        caps: min max values of the explored area. Right now: if not None, the minimum and maximum values of the
        target_class: target class of the CF to be generated. If None, the algorithm will look for any CF that is predicted to belong to a different class than obs_to_interprete
        n_in_layer: number of observations to generate at each step # to do
        layer_shape: shape of the layer to explore the space
        first_radius: radius of the first hyperball generated to explore the space
        dicrease_radius: parameter controlling the size of the are covered at each step
        sparse: controls the sparsity of the final solution (boolean)
        verbose: text
        """
        def __init__(self,
                    obs_to_interprete,
                    prediction_fn,
                    target_class=None,
                    caps=None,
                    n_in_layer=2000,
                    layer_shape='ring',
                    first_radius=0.1,
                    dicrease_radius=10,
                    sparse=True,
                    verbose=False):
            """
            """
            self.obs_to_interprete = obs_to_interprete
            self.prediction_fn = prediction_fn
            self.y_obs = prediction_fn(obs_to_interprete.reshape(1, -1))
    
            #if target_class == None: #To change: works only for binary classification...
            #target_class = 1 - self.y_obs
    
            self.target_class = target_class
            self.caps = caps
            self.n_in_layer = n_in_layer
            self.first_radius = first_radius
            if dicrease_radius <= 1.0:
                raise ValueError("Parameter dicrease_radius must be > 1.0")
            else:
                self.dicrease_radius = dicrease_radius
    
            self.sparse = sparse
            if layer_shape in ['ring', 'ball', 'sphere']:
                self.layer_shape = layer_shape
            else:
                raise ValueError("Parameter layer_shape must be either 'ring', 'ball' or 'sphere'.")
    
            self.verbose = verbose
    
            if int(self.y_obs) != self.y_obs:
                raise ValueError("Prediction function should return a class (integer)")
    
    
        def find_counterfactual(self):
            """
            Finds the decision border then perform projections to make the explanation sparse.
            """
            ennemies_ = self.exploration()
            closest_ennemy_ = sorted(ennemies_,
                                     key= lambda x: pairwise_distances(self.obs_to_interprete.reshape(1, -1), x.reshape(1, -1)))[0]
            self.e_star = closest_ennemy_
            if self.sparse == True:
                out = self.feature_selection(closest_ennemy_)
            else:
                out = closest_ennemy_
            return out
    
    
        def exploration(self):
            """
            Exploration of the feature space to find the decision boundary. Generation of instances in growing hyperspherical layers.
            """
            n_ennemies_ = 999
            radius_ = self.first_radius
    
            while n_ennemies_ > 0:
    
                first_layer_ = self.ennemies_in_layer_(radius=radius_, caps=self.caps, n=self.n_in_layer, first_layer=True)
                
                n_ennemies_ = first_layer_.shape[0]
                radius_ = radius_ / self.dicrease_radius # radius gets dicreased no matter, even if no enemy?
    
                if self.verbose == True:
                    print("%d ennemies found in initial hyperball."%n_ennemies_)
    
                    if n_ennemies_ > 0:
                        print("Zooming in...")
            else:
                if self.verbose == True:
                    print("Expanding hypersphere...")
    
                iteration = 0
                step_ = radius_ / self.dicrease_radius
                #step_ = (self.dicrease_radius - 1) * radius_/5.0  #To do: work on a heuristic for these parameters
    
                while n_ennemies_ <= 0:
    
                    layer = self.ennemies_in_layer_(layer_shape=self.layer_shape, radius=radius_, step=step_, caps=self.caps,
                                                    n=self.n_in_layer, first_layer=False)
    
                    n_ennemies_ = layer.shape[0]
                    radius_ = radius_ + step_
                    iteration += 1
    
                if self.verbose == True:
                    print("Final number of iterations: ", iteration)
    
            if self.verbose == True:
                print("Final radius: ", (radius_ - step_, radius_))
                print("Final number of ennemies: ", n_ennemies_)
            return layer
    
    
        def ennemies_in_layer_(self, layer_shape='ring', radius=None, step=None, caps=None, n=1000, first_layer=False):
            """
            Basis for GS: generates a hypersphere layer, labels it with the blackbox and returns the instances that are predicted to belong to the target class.
            """
            # todo: split generate and get_enemies
    
            if first_layer:
                layer = generate_ball(self.obs_to_interprete, radius, n)
    
            else:
    
                if self.layer_shape == 'ring':
                    segment = (radius, radius + step)
                    layer = generate_ring(self.obs_to_interprete, segment, n)
    
                elif self.layer_shape == 'sphere':
                    layer = generate_sphere(self.obs_to_interprete, radius + step, n)
    
                elif self.layer_shape == 'ball':
                    layer = generate_ball(self.obs_to_interprete, radius + step, n)
    
            #cap here: not optimal - To do
            if caps != None:
                cap_fn_ = lambda x: min(max(x, caps[0]), caps[1])
                layer = np.vectorize(cap_fn_)(layer)
    
            preds_ = self.prediction_fn(layer)
    
            if self.target_class == None:
                enemies_layer = layer[np.where(preds_ != self.y_obs)]
            else:
                enemies_layer = layer[np.where(preds_ == self.target_class)]
    
            return enemies_layer
    
    
        def feature_selection(self, counterfactual):
            """
            Projection step of the GS algorithm. Make projections to make (e* - obs_to_interprete) sparse. Heuristic: sort the coordinates of np.abs(e* - obs_to_interprete) in ascending order and project as long as it does not change the predicted class
    
            Inputs:
            counterfactual: e*
            """
            if self.verbose == True:
                print("Feature selection...")
    
            move_sorted = sorted(enumerate(abs(counterfactual - self.obs_to_interprete.flatten())), key=lambda x: x[1])
            move_sorted = [x[0] for x in move_sorted if x[1] > 0.0]
    
            out = counterfactual.copy()
    
            reduced = 0
    
            for k in move_sorted:
    
                new_enn = out.copy()
                new_enn[k] = self.obs_to_interprete.flatten()[k]
    
                if self.target_class == None:
                    condition_class = self.prediction_fn(new_enn.reshape(1, -1)) != self.y_obs
    
                else:
                    condition_class = self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class
    
                if condition_class:
                    out[k] = new_enn[k]
                    reduced += 1
    
            if self.verbose == True:
                print("Reduced %d coordinates"%reduced)
            return out
    
    
        def feature_selection_all(self, counterfactual):
            """
            Try all possible combinations of projections to make the explanation as sparse as possible.
            Warning: really long!
            """
            if self.verbose == True:
                print("Grid search for projections...")
            for k in range(self.obs_to_interprete.size):
                print('==========', k, '==========')
                for combo in combinations(range(self.obs_to_interprete.size), k):
                    out = counterfactual.copy()
                    new_enn = out.copy()
                    for v in combo:
                        new_enn[v] = self.obs_to_interprete[v]
                    if self.prediction_fn(new_enn.reshape(1, -1)) == self.target_class:
                        print('bim')
                        out = new_enn.copy()
                        reduced = k
            if self.verbose == True:
                print("Reduced %d coordinates"%reduced)
            return out
    
    class CounterfactualExplanation:
        """
        Class for defining a Counterfactual Explanation: this class will help point to specific counterfactual approaches
        """
        def __init__(self, obs_to_interprete, prediction_fn, method='GS', target_class=None, random_state=None):
            """
            Init function
            method: algorithm to use
            random_state
            """
            self.obs_to_interprete = obs_to_interprete
            self.prediction_fn = prediction_fn
            self.method = method
            self.target_class = target_class
            self.random_state = check_random_state(random_state)
            '''
            self.methods_ = {'GS': growingspheres.GrowingSpheres,
                             #'HCLS': lash.HCLS,
                             #'directed_gs': growingspheres.DirectedGrowingSpheres
                            }'''
            self.methods_ = {'GS': GrowingSpheres,
                    #'HCLS': lash.HCLS,
                    #'directed_gs': growingspheres.DirectedGrowingSpheres
                  }
            self.fitted = 0
    
        def fit(self, caps=None, n_in_layer=2000, layer_shape='ball', first_radius=0.1, dicrease_radius=10, sparse=True, verbose=False):
            """
            find the counterfactual with the specified method
            """
            cf = self.methods_[self.method](self.obs_to_interprete,
                    self.prediction_fn,
                    self.target_class,
                    caps,
                    n_in_layer,
                    layer_shape,
                    first_radius,
                    dicrease_radius,
                    sparse,
                    verbose)
            self.enemy = cf.find_counterfactual()
            self.e_star = cf.e_star
            self.move = self.enemy - self.obs_to_interprete
            self.fitted = 1
    
        def distances(self, metrics=None):
            """
            scores de distances entre l'obs et le counterfactual
            """
            if self.fitted < 1:
                raise AttributeError('CounterfactualExplanation has to be fitted first!')
            return get_distances(self.obs_to_interprete, self.enemy)
    
    
    '''
    X,y = datasets.make_moons(n_samples = 200, shuffle=True, noise=0.05, random_state=0)
    X = (X.copy() - X.mean(axis=0))/X.std(axis=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    clf = SVC(gamma=1, probability=True)
    clf = clf.fit(X_train, y_train)
    print(' ### Accuracy:', accuracy_score(y_test, clf.predict(X_test)))
    '''
    #clf = svm.fit(X_train, y_train)
    #clf = xgboost.fit(X_train, y_train)
    
    
    def plot_classification_contour(X, clf, ax=[0,1]):
        ## Inspired by scikit-learn documentation
        h = .02  # step size in the mesh
        cm = plt.cm.RdBu
        x_min, x_max = X[:, ax[0]].min() - .5, X[:, ax[0]].max() + .5
        y_min, y_max = X[:, ax[1]].min() - .5, X[:, ax[1]].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=.5, cmap=cm)
       
    cfs = pd.DataFrame(columns=X.columns)
    
    for i in range(len(X_test)):
        print("\nRODANDO PASSO {}".format(i+1))
        try:
            #idx = np.random.randint(X_test.shape[0])
            ind = i
            #obs = X_test[idx, :].reshape(1, -1)
            obs = X_test.iloc[ind,:].to_numpy().reshape(1,-1)  
            #CF = cf.CounterfactualExplanation(obs, clf.predict, method='GS')
            CF = CounterfactualExplanation(obs, ml_model.predict, method='GS')
            #CF.fit(n_in_layer=200, first_radius=1.1, dicrease_radius=2.0, sparse=True, verbose=False)
            CF.fit(sparse=True, verbose=False)
            cf_x = CF.enemy.reshape(1, -1)[0]
            cfs.loc[len(cfs)] = cf_x
        except:
            cfs.loc[len(cfs)] = np.nan
    
    X_test_aux = X_test.copy()
    cfs_aux = cfs.copy()
    X_test = pd.DataFrame(scaler_gsg.inverse_transform(X_test),columns = X.columns)
    cfs = pd.DataFrame(scaler_gsg.inverse_transform(cfs),columns = X.columns)   
    
    ''' 
    cfs_target = y_test.reset_index(drop=True).copy()
    target = outcome
    for i in range(len(cfs_target)):
        if cfs_target[target][i] == 1:
            cfs_target[target][i] = 0
        else:
            cfs_target[target][i] = 1
    cfs2 = pd.concat([cfs,cfs_target],axis=1)
    #cfs2.to_csv('C:\\Users\\Miguel\\Documents\\Programacao\\py\\dissertacao\\resultados\\dataset\\gsg_spam_df.csv')
        '''
    cfs_target = y_test.reset_index(drop=True).copy()
    #target = outcome
    for i in range(len(cfs_target)):
        if cfs_target[i] == 1:
            cfs_target[i] = 0
        else:
            cfs_target[i] = 1
    cfs2 = pd.concat([cfs,cfs_target],join='inner',axis=1)
    #cfs2.to_csv('C:\\Users\\Miguel\\Documents\\Programacao\\py\\dissertacao\\resultados\\dataset\\gsg_spam_df.csv')
    
    #(delta,change)= result(X_test, cfs2, continuous)
    (delta,delta_o,change)= result(X_test, cfs, continuous)

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
    #X_cfs=df_cfs[col]
        
    #x_test2 = x_test.reset_index(drop=True)
    
    avaliacao(cfs,X_test,df_cfs,continuous)

    
    return (delta,delta_o,change,df_cfs)
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

X_s2 = df_s.drop(columns = outcome_s)
y_s2 = df_s[outcome_s]

X_a2 = df_a.drop(columns = outcome_a)
y_a2 = df_a[outcome_a]

X_d2 = df_d.drop(columns = outcome_d)
y_d2 = df_d[outcome_d]

X_b2 = df_b.drop(columns = outcome_b)
y_b2 = df_b[outcome_b]


#ml_model = xgb.XGBClassifier()


(delta_s,delta_so,change_s,cfs_s) = gsg_auto (X_s2,y_s2,continuous_s,outcome_s,xgb.XGBClassifier())
#(delta_a,delta_ao,change_a,cfs_a) = gsg_auto (X_a2,y_a2,continuous_a,outcome_a,xgb.XGBClassifier())
#(delta_d,delta_do,change_d,cfs_d) = gsg_auto (X_d2,y_d2,continuous_d,outcome_d,xgb.XGBClassifier())
#(delta_b,delta_bo,change_b,cfs_b) = gsg_auto (X_b2,y_b2,continuous_b,outcome_b,xgb.XGBClassifier())

#delta_sr,delta_sro,change_sr,cfs_sr) = gsg_auto (X_s2,y_s2,continuous_s,outcome_s,RandomForestClassifier())
#(delta_sa,delta_sao,change_sa,cfs_sa) = gsg_auto (X_s2,y_s2,continuous_s,outcome_s,AdaBoostClassifier())


#%%

def arredondar(X,df,continuous):
    for col in X.columns:
        if col not in continuous:
            df[col] = df[col].round()
    
    return df

#arredondar(X_a2,cfs_a,continuous_a)
arredondar(X_s2,cfs_s,continuous_s)
#arredondar(X_s2,cfs_sr,continuous_s)
#arredondar(X_s2,cfs_sa,continuous_s)
#arredondar(X_b2,cfs_b,continuous_b)


#%%INVERSE
#FOI NECESSÁRIO ARREDONDAR
#cf_sr attribute 4 não converteu
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
cfs_b['y'] = le_b10.inverse_transform(cfs_b['y'].astype(int)) #1 Male 0 Female''
'''
#%%
#cfs_a.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\cf_gsg_adult.csv',sep=';',decimal=',')
cfs_s.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\cf_gsg_statlog.csv',sep=';',decimal=',')
#cfs_d.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\cf_gsg_default.csv',sep=';',decimal=',')
#cfs_b.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\cf_gsg_bank.csv',sep=';',decimal=',')
#cfs_sr.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\cf_gsg_statlog_rf.csv',sep=';',decimal=',')
#cfs_sa.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\cf_gsg_statlog_ada.csv',sep=';',decimal=',')

#delta_a.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_gsg_adult.csv',sep=';',decimal=',')
delta_s.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_gsg_statlog.csv',sep=';',decimal=',')
#delta_d.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_gsg_default.csv',sep=';',decimal=',')
#delta_b.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_gsg_bank.csv',sep=';',decimal=',')
#delta_sr.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_gsg_statlog_rf.csv',sep=';',decimal=',')
#delta_sa.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_gsg_statlog_ada.csv',sep=';',decimal=',')

#delta_ao.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_o_gsg_adult.csv',sep=';',decimal=',')
delta_so.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_o_gsg_statlog.csv',sep=';',decimal=',')
#delta_do.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_o_gsg_default.csv',sep=';',decimal=',')
#delta_bo.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_o_gsg_bank.csv',sep=';',decimal=',')
#delta_sro.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_o_gsg_statlog_rf.csv',sep=';',decimal=',')
#delta_sao.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\delta_o_gsg_statlog_ada.csv',sep=';',decimal=',')

#change_a.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\change_gsg_adult.csv',sep=';',decimal=',')
change_s.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\change_gsg_statlog.csv',sep=';',decimal=',')
#change_d.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\change_gsg_default.csv',sep=';',decimal=',')
#change_b.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\change_gsg_bank.csv',sep=';',decimal=',')
#change_sr.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\change_gsg_statlog_rf.csv',sep=';',decimal=',')
#change_sa.to_csv('C:\\Users\\Miguel\\Documents\\MIGUEL\\DISSERTACAO\\ETICA\\resultados\\GSG\\change_gsg_statlog_ada.csv',sep=';',decimal=',')

print('FIM DO PROCESSO')