###############################################################################
# ADINs - base original sem covariáveis [C] - dim 1
descarte_votantes = True
###############################################################################
import sys
import os
import pandas as pd
import numpy as np
from copy import copy
import seaborn as sns
import matplotlib.pyplot as plt

diretorio = r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4'

# Leitura da base
caminho = os.path.join(diretorio,'0.Dados','D01. ADINs - Jeferson Mariano Jurisdicao_constitucional_no_Brasil_1966.xlsx')
AA = pd.read_excel(caminho,sheet_name = 'ADIns')

# Filtros básicos
BB = AA \
    .loc[(AA['Julgamento_resultado'] != 'Aguardando') & # Garantindo que o julgamento ocorreu
         (AA['Julgamento_votação'] == 'Maioria')] # Garantindo que não houve unanimidade
BB = BB.reset_index(drop=True)

# Manipulações para chegar no formato PADRÃO
BB = copy(BB)[[
    'ID_STF','Autora_entidade',
    'Aldir_Passarinho2', 'Alexandre_de_Moraes2', 'Ayres_Britto2',
    'Carlos_Madeira2', 'Carlos_Velloso2', 'Cármen_Lúcia2',
    'Célio_Borja2', 'Celso_de_Mello2',	'Cezar_Peluso2',
    'Dias_Toffoli2', 'Djaci_Falcão2', 'Edson_Fachin2',
    'Ellen_Gracie2', 'Eros_Grau2',	'Francisco_Rezek2',
    'Gilmar_Mendes2', 'Ilmar_Galvão2', 'Joaquim_Barbosa2',
    'Luiz_Fux2', 'Marco_Aurélio2', 'Maurício_Corrêa2',
    'Menezes_Direito2', 'Moreira_Alves2', 'Nelson_Jobim2',
    'Néri_da_Silveira2', 'Octávio_Gallotti2', 'Oscar_Corrêa2',
    'Paulo_Brossard2',	'Rafael_Mayer2', 'Ricardo_Lewandowski2',
    'Roberto_Barroso2', 'Rosa_Weber2', 'Sepúlveda_Pertence2',
    'Sydney_Sanches2', 'Teori_Zavascki2'
]]
CC = pd.melt(BB, id_vars=['ID_STF','Autora_entidade'])
CC['ID_STF'] = 'ID_STF_' + CC['ID_STF'].astype(str)
conditions = [
    (CC['value'] == 'Vencedor(a)'),
    (CC['value'] == 'DERROTADO(A)'),
    True
]
choices = [1,0,np.nan]
CC['value'] = np.select(conditions, choices)
DD = CC[~np.isnan(CC.value)]\
    .reset_index(drop=True)\
    .rename(columns={'variable':'Votante','ID_STF':'Votacao','value':'Voto'})
# Descarte de votantes com menos de 20 votos
if descarte_votantes:
    votantes_mantidos = DD.groupby(['Votante']).size().to_frame('qtd').reset_index().query('''qtd >= 20''')['Votante']
    DD = DD[DD['Votante'].isin(votantes_mantidos)]

# Converte de Descricao para ID
os.chdir(r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4/2.Python')
execfile('Funcoes_DE_PARA.py')
f_cria_DE_PARA(dados=DD,
               nomes_colunas=['Votante','Votacao','Autora_entidade'],
               path=os.path.join(diretorio,'3.Auxiliares','D01. CD DE_PARA.csv'))
EE = f_aplica_DE_PARA(dados=DD,
                      path=os.path.join(diretorio,'3.Auxiliares','D01. CD DE_PARA.csv'))

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
    'dim_latente': [1,1],
    'exp_p': [1,1],
    'regpar_eff': [np.log10(1e-7),np.log10(1e-3)],
    'regpar_latU': [np.log10(1e-7),np.log10(1e-3)],
    'regpar_latP': [np.log10(1e-7),np.log10(1e-3)],
    'paciencia': [2,10],
    'validation_split': [0.10,0.10],
    'expoente_batch_size':[5,13]
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
{'target': -0.43158797469892407,
 'params': {'dim_latente': 1.0,
  'exp_p': 1.0,
  'expoente_batch_size': 5.1463062187535344,
  'learning_rate': -1.8007413657740463,
  'paciencia': 9.910888711251957,
  'regpar_eff': -4.007337382480642,
  'regpar_latP': -5.878224031742379,
  'regpar_latU': -3.842882686194046,
  'validation_split': 0.1}}
'''

# Fixando a semente para garantir replicabilidade
tf.random.set_seed(0)
os.environ['PYTHONHASHSEED'] = '0'
rn.seed(1)
np.random.seed(1)

# Aplicando modelo mais adequado
modelo = f_constroi_arquitetura(
    dados=dados_full,dim_latente=1,exp_p=1,reg_type='L1',
    regpar_eff=10**(-4.007337382480642),regpar_latU=10**(-3.842882686194046),regpar_latP=10**(-5.878224031742379),
    variaveis=variaveis,learning_rate=10**(-1.8007413657740463))
modelo = f_treina_modelo(
    modelo=modelo,dados=dados_train,exp_p=1,variaveis=variaveis,
    paciencia=10,ini_epoch=0,add_epoch=100,validation_split=0.1,
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
