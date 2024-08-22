# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 20:54:21 2023

@author: Maurício
"""

import sys
import os
import pandas as pd
import numpy as np
from copy import copy

diretorio = r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V3'

# Leitura da base
caminho = os.path.join(diretorio,'0.Dados','D02. VOTOS2.xlsx')
AA = pd.read_excel(caminho,sheet_name='Rps-julgadas')

# Filtros básicos
BB = AA\
    .loc[AA['VOTACAO'] == 'POR MAIORIA']\
    .drop(columns=['RP','DATA_INGRESSO','DATA_SAIDA','COMPOSICAO','VOTOS'])\
    .reset_index(drop=True)

# Manipulações para chegar no formato PADRÃO
CC = pd.melt(BB, id_vars=['ID','UF','Relator','TEMA','AMBITO','RECREQ','RESULTADO','VOTACAO'])
CC['ID_STF'] = 'ID_STF_' + CC['ID'].astype(str)
conditions = [
    (CC['value'] == 'NI'),
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

# Converte de Descricao para ID
sys.path.insert(1,r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V3/2.Python')
from Funcoes_DE_PARA import *
f_cria_DE_PARA(dados=DD,
               nomes_colunas=['Votante','Votacao','UF','Relator','TEMA','AMBITO','RECREQ','Presidente','Magistratura','Politica'],
               path=os.path.join(diretorio,'3.Auxiliares','D02. DE_PARA.csv'))
EE = f_aplica_DE_PARA(dados=DD,
                      path=os.path.join(diretorio,'3.Auxiliares','D02. DE_PARA.csv'))

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
sys.path.insert(1,r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V3/2.Python')
from Funcoes_MODELAGEM import *
import bayes_opt
import tensorflow as tf
from itertools import chain
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1,l2
from sklearn.metrics import roc_curve,auc,f1_score
import random as rn

# Determinação do modelo
var_eff_num = ['intercepto']
var_eff_cat = ['UF','Relator','TEMA','AMBITO','RECREQ','Presidente','Magistratura','Politica']
 # Votante
var_latU_num = []
var_latU_cat = ['Presidente','Magistratura','Politica']
# Votacao
var_latP_num = []
var_latP_cat = ['UF','Relator','TEMA','AMBITO','RECREQ']
variaveis = [var_eff_num,var_eff_cat,var_latU_num,var_latU_cat,var_latP_num,var_latP_cat]

# Função para otimização bayesiana
def f_treino_CV(
        learning_rate,
        dim_latente,
        exp_p,
        regpar_eff,
        regpar_latU,
        regpar_latP,
        paciencia,
        validation_split,
        expoente_batch_size
        ):
    
    dim_latente = int(np.round(dim_latente))
    exp_p = int(np.round(exp_p))
    paciencia = int(np.round(paciencia))
    expoente_batch_size = int(np.round(expoente_batch_size))
    learning_rate = 10**(learning_rate)
    regpar_eff = 10**(regpar_eff)
    regpar_latU = 10**(regpar_latU)
    regpar_latP = 10**(regpar_latP)
    
    tf.random.set_seed(0)
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(1)
    np.random.seed(1)

    performance = np.repeat(np.nan,nfold)
    for i in range(nfold):
        modelo = f_constroi_arquitetura(
            # Fixos
            dados=dados_full,
            reg_type='L1',
            variaveis=variaveis,
            # Otimizáveis
            dim_latente=dim_latente,
            exp_p=exp_p,
            learning_rate=learning_rate,
            regpar_eff=regpar_eff,
            regpar_latU=regpar_latU,
            regpar_latP=regpar_latP
            )
        modelo = f_treina_modelo(
            # Fixos
            dados=dados_train.loc[dados_train['dev'] != i],
            modelo=modelo,
            verbose=False,
            ini_epoch=0,
            add_epoch=500,
            variaveis=variaveis,
            # Otimizáveis
            exp_p=exp_p,
            paciencia=paciencia,
            validation_split=validation_split,
            batch_size=2**expoente_batch_size
            )
        y_pred,y_true = f_predicao_modelo(
            modelo=modelo,
            dados=dados_train.loc[dados_train['dev'] == i],
            exp_p=exp_p,
            variaveis=variaveis
            )
        performance[i] = BinaryCrossEntropy(y_true=np.array(y_pred),
                                            y_pred=np.array(y_pred))
    return -np.mean(performance)

# Determinação dos grids de hiperparâmetros
bounds = {
    'learning_rate': [np.log10(5e-4),np.log10(5e-2)],
    'dim_latente': [1,1],
    'exp_p': [1,1],
    'regpar_eff': [np.log10(1e-7),np.log10(1e-1)],
    'regpar_latU': [np.log10(1e-7),np.log10(1e-1)],
    'regpar_latP': [np.log10(1e-7),np.log10(1e-1)],
    'paciencia': [1,20],
    'validation_split': [0.05,0.15],
    'expoente_batch_size':[4,13]
    }
optimizer = bayes_opt.BayesianOptimization(f=f_treino_CV,
                                           pbounds=bounds,
                                           verbose=2,
                                           random_state=1)
acquisition_function = bayes_opt.UtilityFunction(kind='ucb',
                                                 kappa=2.546,
                                                 kappa_decay=1)
optimizer.maximize(init_points=25,n_iter=75,acquisition_function=acquisition_function)

optimizer.max

### Retreina apenas o modelo final
# caminho = os.path.join(diretorio,'3.Auxiliares','D02_Modelo_0dim.py')
# caminho = os.path.join(diretorio,'3.Auxiliares','D02_Modelo_1dim.py')
# caminho = os.path.join(diretorio,'3.Auxiliares','D02_Modelo_1dim_covar.py')
# caminho = os.path.join(diretorio,'3.Auxiliares','D02_Modelo_2dim.py')
caminho = os.path.join(diretorio,'3.Auxiliares','D02_Modelo_3dim.py')
execfile(caminho)

############################
# Avaliação de performance #
############################
# AUC
y_pred,y_true = f_predicao_modelo(modelo=modelo,dados=dados_valid,exp_p=1,variaveis=variaveis)
fpr, tpr, thresholds = roc_curve(np.array(y_true),np.array(y_pred))
auc(fpr, tpr)
# Acurácia
melhor_acuracia = 0
melhor_corte = 0.01
for corte in np.linspace(start = 0.01, stop = 0.99, num = 99).tolist():
    pred_valid = np.where(pd.Series(y_pred) < corte,0,1)
    acuracia_corrente = np.mean(pred_valid == y_true)
    if acuracia_corrente > melhor_acuracia:
        melhor_acuracia = acuracia_corrente
        melhor_corte = corte
print(melhor_acuracia)
# F1
pred_valid = np.where(pd.Series(y_pred) < melhor_corte,0,1)
f1_score(y_true, pred_valid)

# Obtendo pesos para cada votante
aux = pd.DataFrame({'Votante': EE['Votante'].unique()}).\
    sort_values(by="Votante")
aux['Votacao'] = 1
aux['Autora_entidade'] = 1
aux['Peso'] = modelo.get_weights()[1]

# Retornando dados de ID para Descrição
descricao = f_aplica_PARA_DE(aux[['Votante','Votacao','Autora_entidade']],
                                  os.path.join(diretorio,'3.Auxiliares','D01. DE_PARA.csv'))
aux['Votante'] = descricao['Votante']
aux['Votante'] = [x[0:-1] for x in aux['Votante']]

# Gráfico com os pesos
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={'figure.figsize':(10,15)})
sns.scatterplot(data=aux,x='Peso',y='Votante',size=5)
