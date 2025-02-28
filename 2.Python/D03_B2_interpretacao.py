###############################################################################
# CamaraDeputados - [B] - dim 2
descarte_wnom = False
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
               path=os.path.join(diretorio,'3.Auxiliares','D03. AB DE_PARA.csv'))
CC = f_aplica_DE_PARA(dados=BB,
                      path=os.path.join(diretorio,'3.Auxiliares','D03. AB DE_PARA.csv'))

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
var_eff_cat = ['Votante','Partido','UF','Votacao']
 # Votante
var_latU_num = []
var_latU_cat = ['Votante','Partido','UF']
# Votacao
var_latP_num = []
var_latP_cat = ['Votacao']
variaveis = [var_eff_num,var_eff_cat,var_latU_num,var_latU_cat,var_latP_num,var_latP_cat]

'''
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

{'target': -0.26297988083250406,
 'params': {'dim_latente': 2.0,
  'exp_p': 1.0,
  'expoente_batch_size': 10.0,
  'learning_rate': -3.3010299956639813,
  'paciencia': 10.0,
  'regpar_eff': -5.326338252400507,
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
    regpar_eff=10**(-5.326338252400507),regpar_latU=10**(-7),regpar_latP=10**(-7),
    variaveis=variaveis,learning_rate=10**(-3.3010299956639813))
modelo = f_treina_modelo(
    modelo=modelo,dados=dados_train,exp_p=1,variaveis=variaveis,
    paciencia=10,ini_epoch=0,add_epoch=100,validation_split=0.1,
    batch_size=2**10,verbose=True)
y_pred,y_true = f_predicao_modelo(modelo=modelo,dados=dados_valid,exp_p=1,variaveis=variaveis)

###############################
### Interpretação dos Pesos ###
###############################

intercepto = modelo.get_weights()[8][0][0]
alpha_votante = [x[0] for x in modelo.get_weights()[0]]
alpha_partido = [x[0] for x in modelo.get_weights()[1]]
alpha_uf = [x[0] for x in modelo.get_weights()[2]]
alpha_votacao = [x[0] for x in modelo.get_weights()[3]]
dim_lat1_votante = [x[0] for x in modelo.get_weights()[4]]
dim_lat2_votante = [x[1] for x in modelo.get_weights()[4]]
dim_lat1_partido = [x[0] for x in modelo.get_weights()[5]]
dim_lat2_partido = [x[1] for x in modelo.get_weights()[5]]
dim_lat1_uf = [x[0] for x in modelo.get_weights()[6]]
dim_lat2_uf = [x[1] for x in modelo.get_weights()[6]]
dim_lat1_votacao = [x[0] for x in modelo.get_weights()[7]]
dim_lat2_votacao = [x[1] for x in modelo.get_weights()[7]]

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
for pos in range(5000):
    i = int(dados_valid.iloc[pos,0])
    j = int(dados_valid.iloc[pos,3])
    k = int(dados_valid.iloc[pos,1])
    l = int(dados_valid.iloc[pos,2])
    efeito_aditivo = intercepto+alpha_votante[i]+alpha_votacao[j]+alpha_partido[k]+alpha_uf[l]
    dim_lat1 = (dim_lat1_votante[i]+dim_lat1_partido[k]+dim_lat1_uf[l])*dim_lat1_votacao[j]
    dim_lat2 = (dim_lat2_votante[i]+dim_lat2_partido[k]+dim_lat2_uf[l])*dim_lat2_votacao[j]
    eta_ijkl = efeito_aditivo + dim_lat1 + dim_lat2
    conta_simples = f_sigmoid(eta_ijkl)
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

### Gera insumos para gráficos no R

# LE DE-PARA
caminho = os.path.join(diretorio,'3.Auxiliares','D03. AB DE_PARA.csv')
depara = pd.read_csv(caminho)

# Votante
depara_aux = depara \
    .query('''Variavel == "Votante"''') \
    .drop(['Variavel'],axis=1) \
    .sort_values(['ID'],ascending=True)
depara_aux['alpha'] = alpha_votante
depara_aux['dim_lat1'] = dim_lat1_votante
depara_aux['dim_lat2'] = dim_lat2_votante
depara_aux.columns = ['id','votante','alpha','dim_lat1','dim_lat2']
# Salva para ser processado no R
caminho_out = os.path.join(diretorio,'3.Auxiliares','D03_B2_Votante.csv')
depara_aux.to_csv(caminho_out,index=False)

# Partido
depara_aux = depara \
    .query('''Variavel == "Partido"''') \
    .drop(['Variavel'],axis=1) \
    .sort_values(['ID'],ascending=True)
depara_aux['alpha'] = alpha_partido
depara_aux['dim_lat1'] = dim_lat1_partido
depara_aux['dim_lat2'] = dim_lat2_partido
depara_aux.columns = ['id','partido','alpha','dim_lat1','dim_lat2']
# Salva para ser processado no R
caminho_out = os.path.join(diretorio,'3.Auxiliares','D03_B2_Partido.csv')
depara_aux.to_csv(caminho_out,index=False)

# UF
depara_aux = depara \
    .query('''Variavel == "UF"''') \
    .drop(['Variavel'],axis=1) \
    .sort_values(['ID'],ascending=True)
depara_aux['alpha'] = alpha_uf
depara_aux['dim_lat1'] = dim_lat1_uf
depara_aux['dim_lat2'] = dim_lat2_uf
depara_aux.columns = ['id','uf','alpha','dim_lat1','dim_lat2']
# Salva para ser processado no R
caminho_out = os.path.join(diretorio,'3.Auxiliares','D03_B2_UF.csv')
depara_aux.to_csv(caminho_out,index=False)

# Votacao
depara_aux = depara \
    .query('''Variavel == "Votacao"''') \
    .drop(['Variavel'],axis=1) \
    .sort_values(['ID'],ascending=True)
depara_aux['alpha'] = alpha_votacao
depara_aux['dim_lat1'] = dim_lat1_votacao
depara_aux['dim_lat2'] = dim_lat2_votacao
depara_aux.columns = ['id','votacao','alpha','dim_lat1','dim_lat2']
# Salva para ser processado no R
caminho_out = os.path.join(diretorio,'3.Auxiliares','D03_B2_Votacao.csv')
depara_aux.to_csv(caminho_out,index=False)

### Gera exemplo para dissertação

# LE DE-PARA
caminho = os.path.join(diretorio,'3.Auxiliares','D03. AB DE_PARA.csv')
depara = pd.read_csv(caminho)

# Escolhe a linha no dados_valid
pos = 0

# Intercepto
intercepto
# 0.1801997

# Votante
i = int(dados_valid.iloc[pos,0])
depara_aux = depara \
    .query('''Variavel == "Votante"''') \
    .sort_values(['ID'],ascending=True)
depara_aux.loc[depara_aux['ID'] == i,'Descricao']
# Ana Catarina
alpha_votante[i]
# -2.3168754e-05

# Votação
j = int(dados_valid.iloc[pos,3])
depara_aux = depara \
    .query('''Variavel == "Votacao"''') \
    .sort_values(['ID'],ascending=True)
depara_aux.loc[depara_aux['ID'] == j,'Descricao']
# 14541-69
alpha_votacao[j]
# -0.0035841505

# Partido
k = int(dados_valid.iloc[pos,1])
depara_aux = depara \
    .query('''Variavel == "Partido"''') \
    .sort_values(['ID'],ascending=True)
depara_aux.loc[depara_aux['ID'] == k,'Descricao']
# PMDB
alpha_partido[k]
# 0.042201176

# UF
l = int(dados_valid.iloc[pos,2])
depara_aux = depara \
    .query('''Variavel == "UF"''') \
    .sort_values(['ID'],ascending=True)
depara_aux.loc[depara_aux['ID'] == l,'Descricao']
# RN
alpha_uf[l]
# 0.08597949

# Efeito Aditivo
efeito_aditivo = intercepto+alpha_votante[i]+alpha_votacao[j]+alpha_partido[k]+alpha_uf[l]
# 0.30477303

# Votante - Coordenada 1
dim_lat1_votante[i]
# 7.473682e-07
dim_lat1_partido[k]
# -1.9152123
dim_lat1_uf[l]
# -1.5528598
dim_lat1_votante[i]+dim_lat1_partido[k]+dim_lat1_uf[l]
# -3.4680715

# Votante - Coordenada 2
dim_lat2_votante[i]
# -0.4386953
dim_lat2_partido[k]
# -2.1518717
dim_lat2_uf[l]
# -0.5678249
dim_lat2_votante[i]+dim_lat2_partido[k]+dim_lat2_uf[l]
# -3.158392

# Votação - Coordenada 1
dim_lat1_votacao[j]
# -0.860429

# Votação - Coordenada 2
dim_lat2_votacao[j]
# -0.7535424

# Dimensão Latente
dim_lat1 = (dim_lat1_votante[i]+dim_lat1_partido[k]+dim_lat1_uf[l])*dim_lat1_votacao[j]
dim_lat2 = (dim_lat2_votante[i]+dim_lat2_partido[k]+dim_lat2_uf[l])*dim_lat2_votacao[j]
dim_lat1+dim_lat2
# 5.364012

# Eta
eta_ijkl = efeito_aditivo + dim_lat1 + dim_lat2
eta_ijkl
# 5.6687846

def f_sigmoid(x):
    a = tf.constant([x], dtype = tf.float64)
    b = tf.keras.activations.sigmoid(a)
    c = b.numpy()[0]
    return (c+1e-15)/(1+2e-15)
conta_simples = f_sigmoid(eta_ijkl)
# 0.9965598174150908

# Valor Real
valor_real = y_true[pos]
valor_real
# 1
