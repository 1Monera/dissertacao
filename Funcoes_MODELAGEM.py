# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:09:04 2023

@author: Maurício
"""

import os
from copy import copy
import pandas as pd
import numpy as np
import tensorflow as tf
from itertools import chain
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1,l2
from sklearn.metrics import roc_curve,auc,f1_score
import keras.backend as K
import seaborn as sns
import random as rn
tf.config.set_visible_devices([],'GPU')
import sys
sys.path.insert(1,r'C:\Users\Maurício\Google Drive\Organizacao\Diversos\Sons Python')
from mario import play_mario

# Função para construir a arquitetura da rede neural
def f_constroi_arquitetura(
        dados,
        dim_latente,
        exp_p,
        reg_type,
        regpar_eff,
        regpar_latU,
        regpar_latP,
        variaveis,
        learning_rate
        ):
    
    if reg_type == 'L1':
        def reg_func(reg_par):
            return l1(reg_par)
    elif reg_type == 'L2':
        def reg_func(reg_par):
            return l2(reg_par)
    
    var_eff_num,var_eff_cat,var_latU_num,var_latU_cat,var_latP_num,var_latP_cat = variaveis
    
    inputs = []
    
    intercepto_input = keras.Input(shape=(1,),dtype='float64')
    intercepto_embed = layers.Embedding(1,1)(intercepto_input)
    intercepto_flatten = layers.Flatten()(intercepto_embed)
    
    inputs.append(intercepto_input)
    
    layers_efeitos_cat = [intercepto_flatten]
    layers_efeitos_num = []
    for variavel in var_eff_num:
        for i in np.arange(1,exp_p+1):
            var_i_name = variavel + '^' + str(i)
            locals()[var_i_name] = keras.Input(shape=(1,),dtype='float64')
            inputs.append(locals()[var_i_name])
            var_ii_name = variavel + '^' + str(i) + '_flatten'
            locals()[var_ii_name] = layers.Flatten()(locals()[var_i_name])
            layers_efeitos_num.append(locals()[var_ii_name])
    for variavel in var_eff_cat:
        n = dados[variavel].unique().shape[0]
        var_name = variavel + '_input'
        var_flatten = variavel + '_flatten'
        locals()[var_name] = keras.Input(shape=(1,),dtype='int64')
        locals()[variavel] = layers.Embedding(n,1,embeddings_regularizer=reg_func(regpar_eff))(locals()[var_name])
        locals()[var_flatten] = layers.Flatten()(locals()[variavel])
        inputs.append(locals()[var_name])
        layers_efeitos_cat.append(locals()[var_flatten])

    concat_efeitos_cat = layers.Concatenate()(layers_efeitos_cat)
    dense_efeitos_cat = layers.Dense(1,activation='linear',kernel_regularizer=reg_func(regpar_eff),kernel_initializer='ones',use_bias=False,trainable=False)(concat_efeitos_cat)
    concat_efeitos_num = layers.Concatenate()(layers_efeitos_num)
    dense_efeitos_num = layers.Dense(1,activation='linear',kernel_regularizer=reg_func(regpar_eff))(concat_efeitos_num)
    eff_add = layers.Add()([dense_efeitos_cat,dense_efeitos_num])
    efeitos = layers.Flatten()(eff_add)
    
    if len(var_latU_num)+len(var_latU_cat) > 0 and len(var_latP_num)+len(var_latP_cat) > 0:
        # Espaço latente U
        flat_latU = []
        if len(var_latU_num) > 0:
            for variavel in var_latU_num:
                aux_intercept = 'intercept_' + variavel
                aux_embed = 'embed_' + variavel
                var_name = 'num_' + variavel
                flat_name = variavel + '_flat'
                locals()[aux_intercept] = keras.Input(shape=(1,),dtype='int64')
                locals()[aux_embed] = layers.Embedding(1,dim_latente,embeddings_regularizer=reg_func(regpar_latU))(locals()[aux_intercept])
                locals()[var_name] = keras.Input(shape=(1,),dtype='float64')
                locals()[flat_name] = layers.Multiply()([locals()[aux_embed],locals()[var_name]])
                inputs.append(locals()[aux_intercept])
                inputs.append(locals()[var_name])
                flat_latU.append(locals()[flat_name])
        if len(var_latU_cat) > 0:
            for variavel in var_latU_cat:
                n = dados[variavel].unique().shape[0]
                var_name = variavel + '_input'
                embed_name = variavel + '_embed'
                flat_name = variavel + '_flat'
                locals()[var_name] = keras.Input(shape=(1,),dtype='int64')
                locals()[embed_name] = layers.Embedding(n,dim_latente,embeddings_regularizer=reg_func(regpar_latU))(locals()[var_name])
                locals()[flat_name] = layers.Flatten()(locals()[embed_name])
                inputs.append(locals()[var_name])
                flat_latU.append(locals()[flat_name])
        latU = layers.Add()(flat_latU)
        # Espaço latente P
        flat_latP = []
        if len(var_latP_num) > 0:
            for variavel in var_latP_num:
                aux_intercept = 'intercept_' + variavel
                aux_embed = 'embed_' + variavel
                var_name = 'num_' + variavel
                flat_name = variavel + '_flat'
                locals()[aux_intercept] = keras.Input(shape=(1,),dtype='int64')
                locals()[aux_embed] = layers.Embedding(1,dim_latente,embeddings_regularizer=reg_func(regpar_latP))(locals()[aux_intercept])
                locals()[var_name] = keras.Input(shape=(1,),dtype='float64')
                locals()[flat_name] = layers.Multiply()([locals()[aux_embed],locals()[var_name]])
                inputs.append(locals()[aux_intercept])
                inputs.append(locals()[var_name])
                flat_latP.append(locals()[flat_name])
        if len(var_latP_cat) > 0:
            for variavel in var_latP_cat:
                n = dados[variavel].unique().shape[0]
                var_name = variavel + '_input'
                embed_name = variavel + '_embed'
                flat_name = variavel + '_flat'
                locals()[var_name] = keras.Input(shape=(1,),dtype='int64')
                locals()[embed_name] = layers.Embedding(n,dim_latente,embeddings_regularizer=reg_func(regpar_latP))(locals()[var_name])
                locals()[flat_name] = layers.Flatten()(locals()[embed_name])
                inputs.append(locals()[var_name])
                flat_latP.append(locals()[flat_name])
        latP = layers.Add()(flat_latP)
        
        dot = layers.Dot(1,normalize=False)([latU,latP])
        dot_flat = layers.Flatten()(dot)
        concat_eff_lat = layers.Concatenate()([efeitos,dot_flat])
        output = layers.Dense(1,activation='sigmoid',kernel_initializer='ones',use_bias=False,trainable=False)(concat_eff_lat)
    else:
        output = layers.Dense(1,activation='sigmoid',kernel_initializer='ones',use_bias=False,trainable=False)(efeitos)
    
    modelo = keras.Model(inputs=inputs,outputs=output)
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    modelo.compile(optimizer='adam',loss='binary_crossentropy')
    
    return modelo

# Função para treinar a rede neural
def f_treina_modelo(
        modelo,
        dados,
        exp_p,
        variaveis,
        paciencia,
        ini_epoch,
        add_epoch,
        validation_split,
        batch_size,
        verbose
        ):
    
    var_eff_num,var_eff_cat,var_latU_num,var_latU_cat,var_latP_num,var_latP_cat = variaveis
    
    dados = copy(dados)
    
    efeitos_num = []
    if len(var_eff_num) > 0:
        for var in var_eff_num:
            for i in np.arange(1,exp_p+1):
                nome_var = var + '^' + str(i)
                aux = pd.DataFrame({nome_var:dados[var]**i})
                dados = pd.concat([dados,aux],axis=1)
                efeitos_num.append(nome_var)
    
    vars_latU_num = []
    for variavel in var_latU_num:
        vars_latU_num.append('intercepto')
        vars_latU_num.append(variavel)
    vars_latP_num = []
    for variavel in var_latP_num:
        vars_latP_num.append('intercepto')
        vars_latP_num.append(variavel)
    
    X_cols = list(chain.from_iterable([
            ['intercepto'],
            var_eff_num,
            var_eff_cat,
            vars_latU_num,
            var_latU_cat,
            vars_latP_num,
            var_latP_cat,
        ]))
    X = [dados[i] for i in X_cols]
    y = dados['target'].to_numpy()
    
    tf.random.set_seed(0)
    os.environ['PYTHONHASHSEED'] = '0'
    rn.seed(1)
    np.random.seed(1)
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',mode='min',patience=paciencia,restore_best_weights=True)
    modelo.fit(
        X,
        y,
        validation_split = validation_split,
        initial_epoch=ini_epoch,
        epochs=ini_epoch+add_epoch,
        use_multiprocessing=False,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=[callback],
        shuffle=True
    )
    return modelo

# Função para predizer a rede neural
def f_predicao_modelo(
        modelo,
        dados,
        exp_p,
        variaveis
        ):
    
    var_eff_num,var_eff_cat,var_latU_num,var_latU_cat,var_latP_num,var_latP_cat = variaveis
    
    dados = copy(dados)
    
    efeitos_num = []
    if len(var_eff_num) > 0:
        for var in var_eff_num:
            for i in np.arange(1,exp_p+1):
                nome_var = var + '^' + str(i)
                aux = pd.DataFrame({nome_var:dados[var]**i})
                dados = pd.concat([dados,aux],axis=1)
                efeitos_num.append(nome_var)
    
    vars_latU_num = []
    for variavel in var_latU_num:
        vars_latU_num.append('intercepto')
        vars_latU_num.append(variavel)
    vars_latP_num = []
    for variavel in var_latP_num:
        vars_latP_num.append('intercepto')
        vars_latP_num.append(variavel)
    
    pred = modelo\
        .predict([dados[x] for x in list(chain.from_iterable([
            ['intercepto'],
            var_eff_num,
            var_eff_cat,
            vars_latU_num,
            var_latU_cat,
            vars_latP_num,
            var_latP_cat,
            ]))],
            verbose=0)
    prob_modelo = [(x[0]+1e-15)/(1+2e-15) for x in pred]
    y_test = [x for x in dados.target]
    return prob_modelo, y_test

# Função para ser maximizada pela otimização bayesiana
def BinaryCrossEntropy(y_true,y_pred):
    y_pred = np.clip(y_pred,1e-7,1-1e-7)
    term_0 = (1-y_true)*np.log(1-y_pred+1e-7)
    term_1 = y_true*np.log(y_pred+1e-7)
    return -np.mean(term_0+term_1,axis=0)

# Função para encontrar as métricas de cada combinação de modelo
def f_parameter_tuning(BB,ghp,verification,variaveis):
    ghp['Accuracy'] = ghp['AUC'] = ghp['F1'] = ghp['Mean_ROC_axis'] = np.nan
    for linha in range(ghp.shape[0]):
        fold,dim,par1,par2,par3,patience,corte = ghp.iloc[linha,0:7]
        fold = str(int(fold))
        dim = int(dim)
        
        if verification == 'Validação':
            dados_train = BB.loc[BB['Fold'+fold] == 'Treino']
            dados_valid = BB.loc[BB['Fold'+fold] == 'Validação']
        elif verification == 'Teste':
            dados_train = BB.loc[BB['Fold'+fold].isin(['Treino','Validação'])]
            dados_valid = BB.loc[BB['Fold'+fold] == 'Teste']
        
        tf.random.set_seed(0)
        os.environ['PYTHONHASHSEED'] = '0'
        rn.seed(1)
        np.random.seed(1)
        
        modelo = f_constroi_arquitetura(BB,dim,1,'L1',par1,par2,par3,variaveis)
        modelo = f_treina_modelo(modelo,dados_train,1,variaveis,patience,0,500,False)
        prob_valid,y_valid = f_predicao_modelo(modelo,dados_valid,1,variaveis)
        
        # Salvando Métricas
        fpr, tpr, thresholds = roc_curve(np.array(y_valid),np.array(prob_valid))
        ghp.loc[linha,'AUC'] = auc(fpr, tpr)
        pred_valid = np.where(pd.Series(prob_valid) < corte,0,1)
        ghp.loc[linha,'Accuracy'] = np.mean(pred_valid == y_valid)
        ghp.loc[linha,'F1'] = f1_score(y_valid, pred_valid)
        sensitivity = np.sum((pd.Series(y_valid) == 1) & (pd.Series(pred_valid) == 1))/np.sum(pd.Series(y_valid) == 1)
        especificity = np.sum((pd.Series(y_valid) == 0) & (pd.Series(pred_valid) == 0))/np.sum(pd.Series(y_valid) == 0)
        ghp.loc[linha,'Mean_ROC_axis'] = (sensitivity+especificity)/2
        print('Linha '+str(linha+1)+'/'+str(ghp.shape[0])+'!')
    play_mario(2)
    return ghp

# Função para encontrar a melhor escolha de hiperparâmetros dentro de um grid
def f_parameter_optimization(BB,grids,variaveis):
    
    grid_dim,grid_reg1,grid_reg2,grid_reg3,grid_pat,grid_cortes = grids
    
    # 1. Busca dimensão
    ghp = [(fold,dim,par1,par2,par3,patience,corte) \
            for fold in range(10) \
            for dim in grid_dim \
            for par1 in [1e-10] \
            for par2 in [1e-10] \
            for par3 in [1e-10] \
            for patience in [2] \
            for corte in [0.5]]
    ghp = pd.DataFrame(ghp)
    ghp.columns = ['fold','dim','par1','par2','par3','patience','corte']
    ghp = f_parameter_tuning(BB,ghp,'Validação',variaveis)
    resumo = ghp.groupby(['dim']).agg({'Accuracy':['mean','std']}).reset_index()
    resumo.columns = ['dim','mean','std']
    resumo['mu-sd'] = resumo['mean'] - resumo['std']
    resumo['mu+sd'] = resumo['mean'] + resumo['std']
    for linha in range(resumo.shape[0]):
        if any(resumo.loc[linha,'mean'] < resumo['mu-sd']):
            resumo = resumo.drop(labels=linha)
    dim0 = resumo.sort_values(['dim'],ascending=True).head(1)['dim'].reset_index().iloc[0,1]
    # sns.pointplot(
    #     x=ghp['dim'].astype(int).astype(str),y=ghp['Accuracy'],
    #     errorbar=("pi",10),capsize=.4,join=False,color="red"
    # )
    print('Etapa 1/6 Concluída')
                       
    # 2. Busca paciência
    ghp = [(fold,dim,par1,par2,par3,patience,corte) \
            for fold in range(10) \
            for dim in [dim0] \
            for par1 in [1e-10] \
            for par2 in [1e-10] \
            for par3 in [1e-10] \
            for patience in grid_pat \
            for corte in [0.5]]
    ghp = pd.DataFrame(ghp)
    ghp.columns = ['fold','dim','par1','par2','par3','patience','corte']
    ghp = f_parameter_tuning(BB,ghp,'Validação',variaveis)
    resumo = ghp.groupby(['patience']).agg({'Accuracy':['mean','std']}).reset_index()
    resumo.columns = ['patience','mean','std']
    resumo['mu-sd'] = resumo['mean'] - resumo['std']
    resumo['mu+sd'] = resumo['mean'] + resumo['std']
    for linha in range(resumo.shape[0]):
        if any(resumo.loc[linha,'mean'] < resumo['mu-sd']):
            resumo = resumo.drop(labels=linha)
    pat0 = resumo.sort_values(['patience'],ascending=True).head(1)['patience'].reset_index().iloc[0,1]
    # sns.pointplot(
    #     x=ghp['patience'].astype(int).astype(str),y=ghp['Accuracy'],
    #     errorbar=("pi",10),capsize=.4,join=False,color="red"
    # )
    print('Etapa 2/6 Concluída')
    
    # 3. Busca par1
    ghp = [(fold,dim,par1,par2,par3,patience,corte) \
            for fold in range(10) \
            for dim in [dim0] \
            for par1 in grid_reg1 \
            for par2 in [1e-10] \
            for par3 in [1e-10] \
            for patience in [pat0] \
            for corte in [0.5]]
    ghp = pd.DataFrame(ghp)
    ghp.columns = ['fold','dim','par1','par2','par3','patience','corte']
    ghp = f_parameter_tuning(BB,ghp,'Validação',variaveis)
    resumo = ghp.groupby(['par1']).agg({'Accuracy':['mean','std']}).reset_index()
    resumo.columns = ['par1','mean','std']
    resumo['mu-sd'] = resumo['mean'] - resumo['std']
    resumo['mu+sd'] = resumo['mean'] + resumo['std']
    for linha in range(resumo.shape[0]):
        if any(resumo.loc[linha,'mean'] < resumo['mu-sd']):
            resumo = resumo.drop(labels=linha)
    par1_0 = resumo.sort_values(['par1'],ascending=False).head(1)['par1'].reset_index().iloc[0,1]
    # sns.pointplot(
    #     x=ghp['par1'].astype(str),y=ghp['Accuracy'],
    #     errorbar=("pi",10),capsize=.4,join=False,color="red"
    # )
    print('Etapa 3/6 Concluída')
    
    # 4. Busca par2,3
    ghp = [(fold,dim,par1,par2,par3,patience,corte) \
            for fold in range(10) \
            for dim in [dim0] \
            for par1 in [par1_0] \
            for par2 in grid_reg2 \
            for par3 in grid_reg3 \
            for patience in [pat0] \
            for corte in [0.5]]
    ghp = pd.DataFrame(ghp)
    ghp.columns = ['fold','dim','par1','par2','par3','patience','corte']
    ghp = f_parameter_tuning(BB,ghp,'Validação',variaveis)
    resumo = ghp.groupby(['par2','par3']).agg({'Accuracy':['mean','std']}).reset_index()
    resumo.columns = ['par2','par3','mean','std']
    resumo['mu-sd'] = resumo['mean'] - resumo['std']
    resumo['mu+sd'] = resumo['mean'] + resumo['std']
    for linha in range(resumo.shape[0]):
        if any(resumo.loc[linha,'mean'] < resumo['mu-sd']):
            resumo = resumo.drop(labels=linha)
    resumo['par2*par3'] = resumo['par2']*resumo['par3']
    par2_0 = resumo.sort_values(['par2*par3'],ascending=False).head(1)['par2'].reset_index().iloc[0,1]
    par3_0 = resumo.sort_values(['par2*par3'],ascending=False).head(1)['par3'].reset_index().iloc[0,1]
    # viz = ghp.groupby(['par2','par3']).agg({'Accuracy':'mean'}).reset_index().pivot('par2','par3','Accuracy')
    # sns.heatmap(viz,annot=True)
    print('Etapa 4/6 Concluída')
    
    # 5. Busca corte
    ghp = [(fold,dim,par1,par2,par3,patience,corte) \
            for fold in range(10) \
            for dim in [dim0] \
            for par1 in [par1_0] \
            for par2 in [par2_0] \
            for par3 in [par3_0] \
            for patience in [pat0] \
            for corte in grid_cortes]
    ghp = pd.DataFrame(ghp)
    ghp.columns = ['fold','dim','par1','par2','par3','patience','corte']
    ghp = f_parameter_tuning(BB,ghp,'Validação',variaveis)
    resumo = ghp.groupby(['corte']).agg({'Accuracy':['mean','std']}).reset_index()
    resumo.columns = ['corte','mean','std']
    resumo['mu-sd'] = resumo['mean'] - resumo['std']
    resumo['mu+sd'] = resumo['mean'] + resumo['std']
    for linha in range(resumo.shape[0]):
        if any(resumo.loc[linha,'mean'] < resumo['mu-sd']):
            resumo = resumo.drop(labels=linha)
    corte_0 = resumo.sort_values(['corte'],ascending=False).head(1)['corte'].reset_index().iloc[0,1]
    print('Etapa 5/6 Concluída')
    
    # 6. Performance final
    ghp = [(fold,dim,par1,par2,par3,patience,corte) \
            for fold in range(10) \
            for dim in [dim0] \
            for par1 in [par1_0] \
            for par2 in [par2_0] \
            for par3 in [par3_0] \
            for patience in [pat0] \
            for corte in [corte_0]]
    ghp = pd.DataFrame(ghp)
    ghp.columns = ['fold','dim','par1','par2','par3','patience','corte']
    ghp = f_parameter_tuning(BB,ghp,'Teste',variaveis)
    resumo = ghp \
                .agg({'Accuracy':['mean','std'],
                      'AUC':['mean','std'],
                      'F1':['mean','std']})
    print(resumo)
    print('Etapa 6/6 Concluída')
    play_mario(3)
    return [dim0,par1_0,par2_0,par3_0,pat0,corte_0]
