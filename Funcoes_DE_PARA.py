# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:50:15 2023

@author: Maurício
"""

from copy import copy
import pandas as pd
import numpy as np

# Função para construir o DE-PARA
def f_cria_DE_PARA(dados,nomes_colunas,path):
    base = copy(dados)
    for i in range(len(nomes_colunas)):
        coluna_corrente = nomes_colunas[i]
        DE_PARA_corrente = pd.DataFrame({'Descricao': base[coluna_corrente].unique()})
        DE_PARA_corrente = DE_PARA_corrente.reset_index().rename(columns={'index':'ID'})
        DE_PARA_corrente['Variavel'] = coluna_corrente
        DE_PARA_corrente = DE_PARA_corrente[['Variavel','ID','Descricao']]
        if i == 0:
            DE_PARA = copy(DE_PARA_corrente)
        else:
            DE_PARA = pd.concat([DE_PARA,DE_PARA_corrente])
    DE_PARA.to_csv(path,index=False)
    return None
# Função para converter de Descricao para ID
def f_aplica_DE_PARA(dados,path):
    dados2 = copy(dados)
    DE_PARA = pd.read_csv(path)
    colunas = DE_PARA['Variavel'].unique()
    for i in range(len(colunas)):
        coluna_corrente = colunas[i]
        DE_PARA_corrente = DE_PARA.loc[DE_PARA['Variavel'] == coluna_corrente]
        dicionario_corrente = dict(zip(DE_PARA_corrente.Descricao,DE_PARA_corrente.ID))
        dados2[coluna_corrente] = dados[coluna_corrente].map(dicionario_corrente)
    return dados2
# Função para converter de ID para Descricao
def f_aplica_PARA_DE(dados,path):
    dados2 = copy(dados)
    DE_PARA = pd.read_csv(path)
    colunas = DE_PARA['Variavel'].unique()
    for i in range(len(colunas)):
        coluna_corrente = colunas[i]
        DE_PARA_corrente = DE_PARA.loc[DE_PARA['Variavel'] == coluna_corrente]
        dicionario_corrente = dict(zip(DE_PARA_corrente.ID,DE_PARA_corrente.Descricao))
        dados2[coluna_corrente] = dados[coluna_corrente].map(dicionario_corrente)
    return dados2