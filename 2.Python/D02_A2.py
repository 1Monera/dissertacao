###############################################################################
# RPs - [A] - dim 2
descarte_wnom = False
###############################################################################
import sys
import os
import pandas as pd
import numpy as np
from copy import copy

diretorio = r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4'

# Leitura da base
caminho = os.path.join(diretorio,'0.Dados','D02. VOTOS2.xlsx')
AA = pd.read_excel(caminho,sheet_name='Rps-julgadas')

# Filtros básicos
BB = AA \
    .loc[AA['VOTACAO'] == 'POR MAIORIA']\
    .drop(columns=['RP','DATA_INGRESSO','DATA_SAIDA','COMPOSICAO','VOTOS'])\
    .reset_index(drop=True)

# Manipulações para chegar no formato PADRÃO
CC = pd.melt(BB, id_vars=['ID','UF','Relator','TEMA','AMBITO','RECREQ','RESULTADO','VOTACAO'])
CC['ID_STF'] = 'ID_STF_' + CC['ID'].astype(str)
conditions = [
    (CC['value'].isin(['NI','AU'])),
    (CC['value'] == CC['RESULTADO']),
    (CC['value'] == CC['RESULTADO'])
]
choices = [np.nan,1,0]
CC['value'] = np.select(conditions, choices)
DD = CC[~np.isnan(CC.value)]\
    .reset_index(drop=True)\
    .rename(columns={'variable':'Votante','ID_STF':'Votacao','value':'Voto'})\
    .drop(columns=['RESULTADO','VOTACAO','ID'])

# Inserindo covariáveis adicionais dos votantes
caminho = os.path.join(diretorio,'0.Dados','D02. VOTOS2.xlsx')
df_votantes = pd.read_excel(caminho,sheet_name = 'ministros')
DD = DD\
    .merge(df_votantes,left_on='Votante',right_on='Ministro',how='left')\
    .drop(columns=['Ministro'])

#caminho = os.path.join(diretorio,'0.Dados','D02. WNOMINATE-FORMAT.csv')
#DD[['Votante','Votacao','Voto']].to_csv(caminho,index=False)

if descarte_wnom:
    caminho = os.path.join(diretorio,'0.Dados','D02. RPs reduzido WNOM.csv')
    base_reduzida = pd.read_csv(caminho)
    # Considera os mesmos votantes que W-NOMINATE
    votantes_wnom = base_reduzida['Votante'].unique()
    DD = DD.loc[DD['Votante'].isin(votantes_wnom)]
    # Considera as mesmas votacoes que W-NOMINATE
    votacoes_wnom = base_reduzida['Votacao'].unique()
    DD = DD.loc[DD['Votacao'].isin(votacoes_wnom)]
    # Reset Índice
    DD = DD.reset_index(drop=True)

# Converte de Descricao para ID
os.chdir(r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4/2.Python')
execfile('Funcoes_DE_PARA.py')
f_cria_DE_PARA(dados=DD,
               nomes_colunas=['Votante','Votacao','UF','Relator','TEMA','AMBITO','RECREQ','Presidente','Magistratura','Politica'],
               path=os.path.join(diretorio,'3.Auxiliares','D02. AB DE_PARA.csv'))
EE = f_aplica_DE_PARA(dados=DD,
                      path=os.path.join(diretorio,'3.Auxiliares','D02. AB DE_PARA.csv'))

# Adiciona intercepto
EE['intercepto'] = 0
# Renomeia 'Voto' para 'target'
EE.rename(columns={'Voto':'target'},inplace=True)

# Transforma todos os valores numéricos em float64
FF = copy(EE)
for coluna in FF.columns:
    FF[coluna] = FF[coluna].astype(np.float64)

# Preparando dados para validação cruzada
np.random.seed(1)
nfold = 10
FF['dev'] = np.random.choice([0,np.nan],size=FF.shape[0],p=[0.9,0.1])
FF.loc[FF['dev'] == 0,'dev'] = np.random.choice(list(range(nfold)),size=FF.loc[FF['dev'] == 0].shape[0])
dados_full = copy(FF)
dados_train = copy(FF.loc[~np.isnan(FF['dev'])])
dados_valid = copy(FF.loc[np.isnan(FF['dev'])])

# Funções modelagem
os.chdir(r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4/2.Python')
execfile('Funcoes_MODELAGEM.py')
import bayes_opt
import tensorflow as tf
from itertools import chain
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1,l2
from sklearn.metrics import roc_curve,auc,f1_score
import random as rn

# Determinação do modelo
var_eff_num = []
var_eff_cat = ['Votante','Votacao']
 # Votante
var_latU_num = []
var_latU_cat = ['Votante']
# Votacao
var_latP_num = []
var_latP_cat = ['Votacao']
variaveis = [var_eff_num,var_eff_cat,var_latU_num,var_latU_cat,var_latP_num,var_latP_cat]

# Determinação dos grids de hiperparâmetros
bounds = {
    'learning_rate': [np.log10(5e-4),np.log10(5e-2)],
    'dim_latente': [2,2],
    'exp_p': [1,1],
    'regpar_eff': [np.log10(1e-7),np.log10(1e-3)],
    'regpar_latU': [np.log10(1e-7),np.log10(1e-3)],
    'regpar_latP': [np.log10(1e-7),np.log10(1e-3)],
    'paciencia': [2,10],
    'validation_split': [0.10,0.10],
    'expoente_batch_size':[5,10]
    }
optimizer = bayes_opt.BayesianOptimization(f=f_treino_CV,
                                           pbounds=bounds,
                                           verbose=2,
                                           random_state=1)
acquisition_function = bayes_opt.UtilityFunction(kind='ucb',
                                                 kappa=2.546)
optimizer.maximize(init_points=10,n_iter=30,acquisition_function=acquisition_function)
optimizer.max
'''
{'target': -0.4967495950130352,
 'params': {'dim_latente': 2.0,
  'exp_p': 1.0,
  'expoente_batch_size': 5.0,
  'learning_rate': -3.3010299956639813,
  'paciencia': 4.726444192252621,
  'regpar_eff': -7.0,
  'regpar_latP': -3.0,
  'regpar_latU': -7.0,
  'validation_split': 0.1}}
'''

# Fixando a semente para garantir replicabilidade
tf.random.set_seed(0)
os.environ['PYTHONHASHSEED'] = '0'
rn.seed(1)
np.random.seed(1)

# Aplicando modelo mais adequado
modelo = f_constroi_arquitetura(
    dados=dados_full,dim_latente=2,exp_p=1,reg_type='L1',
    regpar_eff=10**(-7),regpar_latU=10**(-7),regpar_latP=10**(-3),
    variaveis=variaveis,learning_rate=10**(-3.3010299956639813))
modelo = f_treina_modelo(
    modelo=modelo,dados=dados_train,exp_p=1,variaveis=variaveis,
    paciencia=5,ini_epoch=0,add_epoch=100,validation_split=0.1,
    batch_size=2**5,verbose=True)
y_pred,y_true = f_predicao_modelo(modelo=modelo,dados=dados_valid,exp_p=1,variaveis=variaveis)
fpr, tpr, thresholds = roc_curve(np.array(y_true),np.array(y_pred))
auc(fpr, tpr)

melhor_acuracia = 0
melhor_corte = 0.01
for corte in np.linspace(start = 0.01, stop = 0.99, num = 99).tolist():
    pred_valid = np.where(pd.Series(y_pred) < corte,0,1)
    acuracia_corrente = np.mean(pred_valid == y_true)
    if acuracia_corrente > melhor_acuracia:
        melhor_acuracia = acuracia_corrente
        melhor_corte = corte
print(melhor_acuracia)
pred_valid = np.where(pd.Series(y_pred) < melhor_corte,0,1)
f1_score(y_true, pred_valid)
