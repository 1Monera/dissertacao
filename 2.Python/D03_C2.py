###############################################################################
# CamaraDeputados - [C] - dim 2
descarte_wnom = True
###############################################################################
import sys
import os
import pandas as pd
import numpy as np
from copy import copy

diretorio = r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4'

# Leitura da base
caminho = os.path.join(diretorio,'0.Dados','D03. Camara Deputados PLEN.csv')
AA = pd.read_csv(caminho,encoding='latin-1')

BB = copy(AA)
BB['voto'] = np.where(BB['voto']=='Sim',1,0)
BB = BB.rename(columns={'deputado_nome':'Votante',
                        'deputado_siglaPartido':'Partido',
                        'deputado_siglaUf':'UF',
                        'idVotacao':'Votacao',
                        'voto':'Voto'})\
        .drop(columns=['Data'])

#caminho = os.path.join(diretorio,'0.Dados','D03. WNOMINATE-FORMAT.csv')
#BB[['Votante','Votacao','Voto']].to_csv(caminho,index=False)

if descarte_wnom:
    caminho = os.path.join(diretorio,'0.Dados','D03. CamaraDeputados reduzido WNOM.csv')
    base_reduzida = pd.read_csv(caminho)
    # Considera os mesmos votantes que W-NOMINATE
    votantes_wnom = base_reduzida['Votante'].unique()
    BB = BB.loc[BB['Votante'].isin(votantes_wnom)]
    # Considera as mesmas votacoes que W-NOMINATE
    votacoes_wnom = base_reduzida['Votacao'].unique()
    BB = BB.loc[BB['Votacao'].isin(votacoes_wnom)]
    # Reset Índice
    BB = BB.reset_index(drop=True)

# Converte de Descricao para ID
os.chdir(r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4/2.Python')
execfile('Funcoes_DE_PARA.py')
f_cria_DE_PARA(dados=BB,
               nomes_colunas=['Votante','Partido','UF','Votacao'],
               path=os.path.join(diretorio,'3.Auxiliares','D03. CD DE_PARA.csv'))
CC = f_aplica_DE_PARA(dados=BB,
                      path=os.path.join(diretorio,'3.Auxiliares','D03. CD DE_PARA.csv'))

# Adiciona intercepto
CC['intercepto'] = 0
# Renomeia 'Voto' para 'target'
CC.rename(columns={'Voto':'target'},inplace=True)

# Transforma todos os valores numéricos em float64
FF = copy(CC)
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
    'expoente_batch_size':[10,21]
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
{'target': -0.31432812332318727,
 'params': {'dim_latente': 2.0,
  'exp_p': 1.0,
  'expoente_batch_size': 10.48491384787587,
  'learning_rate': -2.3906666307986324,
  'paciencia': 2.0,
  'regpar_eff': -4.293188548454206,
  'regpar_latP': -7.0,
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
    regpar_eff=10**(-4.293188548454206),regpar_latU=10**(-7),regpar_latP=10**(-7),
    variaveis=variaveis,learning_rate=10**(-2.3906666307986324))
modelo = f_treina_modelo(
    modelo=modelo,dados=dados_train,exp_p=1,variaveis=variaveis,
    paciencia=2,ini_epoch=0,add_epoch=100,validation_split=0.1,
    batch_size=2**10,verbose=True)
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
