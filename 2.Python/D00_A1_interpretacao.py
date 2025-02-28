###############################################################################
# sen90 - base original sem covariáveis [A] - dim 1
###############################################################################
import sys
import os
import pandas as pd
import numpy as np
from copy import copy
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

diretorio = r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4'

# Leitura da base
caminho = os.path.join(diretorio,'0.Dados','D00. Original.csv')
AA = pd.read_csv(caminho)

# Manipulações para chegar no formato PADRÃO
AA.rename(columns={'Unnamed: 0':'Votante'},inplace=True)
BB = pd.melt(AA, id_vars=['Votante'])
conditions = [
    (BB['value'].ge(1)) & (BB['value'].le(3)),
    (BB['value'].ge(4)) & (BB['value'].le(6)),
    (BB['value'].ge(7)) & (BB['value'].le(9)),
    True,
]
choices = [1,0,-1,-2]
BB['value'] = np.select(conditions, choices)
# Gráfico com proporção de cada tipo de voto
BB_aux = BB \
    .groupby(['value']) \
    .agg(qtd=('value','count')) \
    .reset_index() \
    .rename(
        columns={
            'value': 'Tipo Voto',
            'qtd': 'Frequência',
        }
    )
conditions = [
    BB_aux['Tipo Voto'] == 1,
    BB_aux['Tipo Voto'] == 0,
    BB_aux['Tipo Voto'] == -1,
    BB_aux['Tipo Voto'] == -2,
]
choices = ['Sim','Não','Abstenção','Fora de legislatura']
BB_aux['Tipo Voto'] = np.select(conditions,choices)
ax = sns.barplot(data=BB_aux,x='Tipo Voto',y='Frequência',estimator="sum",errorbar=None)
caminho = os.path.join(diretorio,'5.Gráficos','D00_Proporcao_Votos_Originais.png')
ax.get_figure().savefig(caminho,dpi=500)
# Removendo votos de abstenção e fora de legislatura
CC = BB.query('value >= 0')
CC.reset_index(drop=True,inplace=True)
CC.rename(columns={'variable':'Votacao','value':'Voto'},inplace=True)
CC['Partido'] = [x[-5] for x in CC['Votante']]
CC.loc[CC['Votante'] == 'JOHNSON (D USA)','Partido'] = 'D'
# Converte de Descricao para ID
os.chdir(r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4/2.Python')
execfile('Funcoes_DE_PARA.py')
f_cria_DE_PARA(dados=CC,
               nomes_colunas=['Votante','Votacao','Partido'],
               path=os.path.join(diretorio,'3.Auxiliares','D00. AB DE_PARA.csv'))
DD = f_aplica_DE_PARA(dados=CC,
                      path=os.path.join(diretorio,'3.Auxiliares','D00. AB DE_PARA.csv'))
# Adiciona intercepto
DD['intercepto'] = 0
# Renomeia 'Voto' para 'target'
DD.rename(columns={'Voto':'target'},inplace=True)
# Transforma todos os valores numéricos em float64
EE = copy(DD)
for coluna in EE.columns:
    EE[coluna] = EE[coluna].astype(np.float64)

# Preparando dados para validação cruzada
np.random.seed(1)
nfold = 10
EE['dev'] = np.random.choice([0,np.nan],size=EE.shape[0],p=[0.9,0.1])
EE.loc[EE['dev'] == 0,'dev'] = np.random.choice(list(range(nfold)),size=EE.loc[EE['dev'] == 0].shape[0])
dados_full = copy(EE)
dados_train = copy(EE.loc[~np.isnan(EE['dev'])])
dados_valid = copy(EE.loc[np.isnan(EE['dev'])])

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

'''
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
    'expoente_batch_size':[5,16]
    }
optimizer = bayes_opt.BayesianOptimization(f=f_treino_CV,
                                           pbounds=bounds,
                                           verbose=1,
                                           random_state=1)
acquisition_function = bayes_opt.UtilityFunction(kind='ucb',
                                                 kappa=2.546)
optimizer.maximize(init_points=10,n_iter=30,acquisition_function=acquisition_function)
optimizer.max

{'target': -0.4156531843034964,
 'params': {'dim_latente': 1.0,
  'exp_p': 1.0,
  'expoente_batch_size': 6.832766993930445,
  'learning_rate': -1.3010299956639813,
  'paciencia': 8.864074504420996,
  'regpar_eff': -7.0,
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
    dados=dados_full,dim_latente=1,exp_p=1,reg_type='L1',
    regpar_eff=10**(-7),regpar_latU=10**(-7),regpar_latP=10**(-7),
    variaveis=variaveis,learning_rate=10**(-1.3010299956639813))
modelo = f_treina_modelo(
    modelo=modelo,dados=dados_train,exp_p=1,variaveis=variaveis,
    paciencia=9,ini_epoch=0,add_epoch=100,validation_split=0.1,
    batch_size=2**7,verbose=True)
y_pred,y_true = f_predicao_modelo(modelo=modelo,dados=dados_valid,exp_p=1,variaveis=variaveis)

###############################
### Interpretação dos Pesos ###
###############################

intercepto = modelo.get_weights()[4][0][0]
alpha_votante = [x[0] for x in modelo.get_weights()[0]]
alpha_votacao = [x[0] for x in modelo.get_weights()[1]]
dim_lat_votante = [x[0] for x in modelo.get_weights()[2]]
dim_lat_votacao = [x[0] for x in modelo.get_weights()[3]]

'''
### TESTA SE OS PESOS FAZEM O QUE SE ESPERA

def f_sigmoid(x):
    a = tf.constant([x], dtype = tf.float64)
    b = tf.keras.activations.sigmoid(a)
    c = b.numpy()[0]
    return (c+1e-15)/(1+2e-15)

teste = pd.DataFrame({
        'pos': [],
        'conta_simples': [],
        'rede_neural': [],
    })
for pos in range(dados_valid.shape[0]):
    i = int(dados_valid.iloc[pos,0])
    j = int(dados_valid.iloc[pos,1])
    eta_ij = intercepto+alpha_votante[i]+alpha_votacao[j]+dim_lat_votante[i]*dim_lat_votacao[j]
    conta_simples = f_sigmoid(eta_ij)
    rede_neural = y_pred[pos]
    corrente = pd.DataFrame({
            'pos': [pos],
            'conta_simples': [conta_simples],
            'rede_neural': [rede_neural],
        })
    teste = pd.concat([teste,corrente],axis=0)
    print(f"Posição {pos}: Conta simples {conta_simples} - Rede Neural {rede_neural}")

ax = sns.scatterplot(teste.sort_values(by=['conta_simples']),x='conta_simples',y='rede_neural',size=0.01,edgecolor=None,legend=False)
ax.plot((1e-3,1e0),(1e-3,1e0),color='r')
plt.show()
'''

# LE DE-PARA
caminho = os.path.join(diretorio,'3.Auxiliares','D00. AB DE_PARA.csv')
depara = pd.read_csv(caminho)

# Votante
depara_aux = depara \
    .query('''Variavel == "Votante"''') \
    .drop(['Variavel'],axis=1) \
    .sort_values(['ID'],ascending=True)
depara_aux.loc[depara_aux['Descricao'] == 'JOHNSON (D USA)','Descricao'] = 'JOHNSON (D US)'
depara_aux['partido'] = [x[-5] for x in depara_aux['Descricao']]
#depara_aux['Descricao'] = [x[0:-7] for x in depara_aux['Descricao']]
depara_aux['alpha'] = alpha_votante
depara_aux['dim_latente'] = dim_lat_votante
depara_aux.columns = ['id','votante','partido','alpha','dim_latente']
# Salva para ser processado no R
caminho_out = os.path.join(diretorio,'3.Auxiliares','D00_A1_Votante.csv')
depara_aux.to_csv(caminho_out,index=False)

# Votacao
depara_aux = depara \
    .query('''Variavel == "Votacao"''') \
    .drop(['Variavel'],axis=1) \
    .sort_values(['ID'],ascending=True)
depara_aux['alpha'] = alpha_votacao
depara_aux['dim_latente'] = dim_lat_votacao
depara_aux.columns = ['id','votacao','alpha','dim_latente']
# Salva para ser processado no R
caminho_out = os.path.join(diretorio,'3.Auxiliares','D00_A1_Votacao.csv')
depara_aux.to_csv(caminho_out,index=False)
