import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import xgboost as xgb

from numpy import mean,sqrt,square
from sklearn.metrics import mean_squared_error
import pickle
import missingno as msno
import auxiliar_functions as af
sns.set_style("darkgrid")
pd.set_option('display.max_columns', None)
pd.options.display.max_colwidth = 100



def pipeline_feature_engineering(df):
      
    df_vars = df.copy()
    
    pipe_df_features=pd.DataFrame()

    pipe_df_features['HH charge cell']=df_vars["HH charge cell"] ##
    pipe_df_features['LL charge cell']=df_vars["LL charge cell"] ##
    pipe_df_features['charge cell']=df_vars["charge cell"] ##

    pipe_df_features["delta HH charge cell"]=pipe_df_features["HH charge cell"]-pipe_df_features["charge cell"] ##
    pipe_df_features["delta LL charge cell"]=pipe_df_features["charge cell"]-pipe_df_features["LL charge cell"] ##

    pipe_df_features["min_water_3"]=df_vars["water"].rolling(window=3).apply(lambda x:np.nanmin(list(x))) # 1
    pipe_df_features["min_solid percentage_10"]=df_vars["solid percentage"].rolling(window=10).apply(lambda x:np.nanmin(list(x))) # 2
    pipe_df_features['HH TPH']=df_vars['HH TPH'] # 3
    pipe_df_features["max_delta LL charge cell_10"]=pipe_df_features["delta LL charge cell"].rolling(window=10).apply(lambda x:np.nanmax(list(x))) # 4
    pipe_df_features['power']=df_vars['power'] # 5
    pipe_df_features["rms_delta LL charge cell_3"]=pipe_df_features["delta LL charge cell"].rolling(window=3).apply(lambda x: sqrt(mean(square(list(x))))) # 6
    pipe_df_features["var_delta HH charge cell_10"]=pipe_df_features["delta HH charge cell"].rolling(window=10).apply(lambda x: np.nanvar(x)) # 7
    pipe_df_features[f"max_covelin law_10"]=df_vars["covelin law"].rolling(window=10).apply(lambda x:np.nanmax(list(x))) # 8
    pipe_df_features[f'LL charge cell_(t-2)'] = pipe_df_features["LL charge cell"].shift(2) # 9
    pipe_df_features[f"min_granulometry_5"]=df_vars["granulometry"].rolling(window=5).apply(lambda x:np.nanmin(list(x))) # 10
    pipe_df_features[f"max_bornite law_10"]=df_vars["bornite law"].rolling(window=10).apply(lambda x:np.nanmax(list(x))) # 11
    pipe_df_features[f"min_charge cell_5"]=pipe_df_features["charge cell"].rolling(window=5).apply(lambda x:np.nanmin(list(x))) # 12
    pipe_df_features[f'chalcocite law_(t-2)'] = df_vars["chalcocite law"].shift(2) # 13
    pipe_df_features[f"max_sag power index_5"]=df_vars["sag power index"].rolling(window=5).apply(lambda x:np.nanmax(list(x))) # 14
    pipe_df_features[f"min_speed_3"]=df_vars["speed"].rolling(window=3).apply(lambda x:np.nanmin(list(x))) # 15
    pipe_df_features[f"var_bornite law_3"]=df_vars["bornite law"].rolling(window=3).apply(lambda x: np.nanvar(x)) # 16
    pipe_df_features[f"var_speed_3"]=df_vars["speed"].rolling(window=3).apply(lambda x: np.nanvar(x)) # 17
    pipe_df_features[f"min_pyrite law_10"]=df_vars["pyrite law"].rolling(window=10).apply(lambda x:np.nanmin(list(x))) # 18
    pipe_df_features[f'crusher index_(t-5)'] = df_vars["crusher index"].shift(5) # 19
    pipe_df_features[f"var_power_3"]=df_vars["power"].rolling(window=3).apply(lambda x: np.nanvar(x)) # 20
    pipe_df_features[f"var_chalcocite law_3"]=df_vars["chalcocite law"].rolling(window=3).apply(lambda x: np.nanvar(x)) # 21
    pipe_df_features[f"var_bornite law_5"]=df_vars["bornite law"].rolling(window=5).apply(lambda x: np.nanvar(x)) # 22
    pipe_df_features[f"var_solid percentage_3"]=df_vars["solid percentage"].rolling(window=3).apply(lambda x: np.nanvar(x)) # 23
    pipe_df_features[f"var_speed_10"]=df_vars["speed"].rolling(window=10).apply(lambda x: np.nanvar(x)) # 24
    pipe_df_features[f"ball work index_(t-1)"] = df_vars["ball work index"].shift(1) # 25
    pipe_df_features[f"var_chalcocite law_5"]=df_vars["chalcocite law"].rolling(window=5).apply(lambda x: np.nanvar(x)) # 26
    pipe_df_features[f"var_water_3"]=df_vars["water"].rolling(window=3).apply(lambda x: np.nanvar(x)) # 27
    pipe_df_features[f'chalcopyrite law_(t-5)'] = df_vars["chalcopyrite law"].shift(5) # 28
    pipe_df_features[f"var_crusher index_10"]=df_vars["crusher index"].rolling(window=10).apply(lambda x: np.nanvar(x)) # 29
    pipe_df_features[f"var_chalcopyrite law_3"]=df_vars["chalcopyrite law"].rolling(window=3).apply(lambda x: np.nanvar(x)) # 30
    pipe_df_features[f"var_granulometry_3"]=df_vars["granulometry"].rolling(window=3).apply(lambda x: np.nanvar(x)) # 31
    pipe_df_features[f"var_delta HH charge cell_3"]=pipe_df_features["delta HH charge cell"].rolling(window=3).apply(lambda x: np.nanvar(x)) # 32
    
    return pipe_df_features

# Select first of vector of atollos cases (0) and append into DF:
def inicio_perdida_tph(DF, variable):
       first_case_atollo = [0]
       DF=DF.copy().reset_index()
       for x in range(1, len(DF)):
              if((DF.loc[DF.index[x], variable] == 1) & (DF.loc[DF.index[x-1], variable] == 0)):
                     first_case_atollo.append(1)
              else:
                     first_case_atollo.append(0)
                     
       return first_case_atollo
      
# Select last of vector of atollos cases (0) and append into DF:
def final_perdida_tph(DF, variable):
       last_case_atollo = []
       DF=DF.copy().reset_index()
       for x in range((len(DF)-1)):
              if((DF.loc[DF.index[x], variable] == 1) & (DF.loc[DF.index[x+1], variable] == 0)):
                     last_case_atollo.append(1)
              else:
                     last_case_atollo.append(0)
       last_case_atollo.append(1) 

       return last_case_atollo

# Create function Atollo_mod: 0 when initial atollo; 2 when not initial atollo; 1 when no atollo:
def loss_tph(DF,variable):
    
    DF['start loss tph']=inicio_perdida_tph(DF=DF, variable=variable)
    DF['end loss tph']=final_perdida_tph(DF=DF, variable=variable)
    #Create Atollo_mod:
    variable_mod = str(variable+'_mod')    
    DF[variable_mod] =  DF[variable].replace(1,2)   
    DF.loc[DF['start loss tph']==1,variable_mod]=1

    return DF

def feature_engineering_recommendation(df):
      
    pipe_df_features=df.copy()
    pipe_df_features["LL charge cell"]=pipe_df_features["HH charge cell"]-80
    pipe_df_features["charge cell"]=(pipe_df_features["HH charge cell"]+pipe_df_features["LL charge cell"])/2 

    pipe_df_features["delta HH charge cell"]=pipe_df_features["HH charge cell"]-pipe_df_features["charge cell"] ##
    pipe_df_features["delta LL charge cell"]=pipe_df_features["charge cell"]-pipe_df_features["LL charge cell"] ##
    
    pipe_df_features["max_delta LL charge cell_10"]=pipe_df_features["delta LL charge cell"].rolling(window=10).apply(lambda x:np.nanmax(list(x))) ##
    pipe_df_features["rms_delta LL charge cell_3"]=pipe_df_features["delta LL charge cell"].rolling(window=3).apply(lambda x: sqrt(mean(square(list(x))))) ##
    pipe_df_features["var_delta HH charge cell_10"]=pipe_df_features["delta HH charge cell"].rolling(window=10).apply(lambda x: np.nanvar(x)) ##
    pipe_df_features[f'LL charge cell_(t-2)'] = pipe_df_features["LL charge cell"].shift(2) ##
    pipe_df_features[f"min_charge cell_5"]=pipe_df_features["charge cell"].rolling(window=5).apply(lambda x:np.nanmin(list(x))) ##
    pipe_df_features[f"var_delta HH charge cell_3"]=pipe_df_features["delta HH charge cell"].rolling(window=3).apply(lambda x: np.nanvar(x)) ##
    
    return pipe_df_features


def tph_function(HH_CC,df,pipe):
    
    df_rec=df.copy()
    df_rec["HH charge cell"]=HH_CC
    df_rec_features=feature_engineering_recommendation(df_rec).iloc[[-1]]
    #display(df_rec_features)

    Ypred=pipe.predict(df_rec_features)[0]
    
    return Ypred

def optimum_recommendation(df,range_hh_cc,pipe):
    
    list_TPH=np.array([tph_function(i,df,pipe) for i in range_hh_cc])
    
    #index_max=np.argmax(list_TPH)

    # Maximo HH charge cell que maximiza TPH
    winner = np.argwhere(list_TPH == np.amax(list_TPH))
    index_max=max(winner.flatten().tolist())

    rec_hh_cc=range_hh_cc[index_max]
    tph_opt=list_TPH[index_max]
    #print("Recommendation: ",rec_hh_cc)
    #print("TPH optimum: ",tph_opt)
    df_tph=pd.DataFrame({"TPH":list_TPH,"HH CC":range_hh_cc})
    
    fig = go.Figure()
#
    fig.add_trace(go.Scatter(x=df_tph["HH CC"], y=df_tph["TPH"],
                        mode='lines+markers',
                        name='TPH'))
#
    fig.update_layout(height=500, width=1200, title_text=f"TPH vs HH charge cell: (Recommendation: {rec_hh_cc} & TPH optimum: {int(tph_opt)})" ,xaxis_title="HH charge cell",
        yaxis_title="TPH")   
#
    fig.update_layout(hovermode="x unified")                
#
    #fig.show()
    
    return rec_hh_cc,tph_opt,fig 