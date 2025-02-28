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
os.chdir(r'C:\Users\Maurício\Google Drive\Organizacao\Diversos\Sons Python')
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
    
    ### Efeitos
    layers_efeitos = []
    
    # Intercepto (Obrigatório)
    var_name = 'intercepto_input'
    var_flatten = 'intercepto_flatten'
    intercepto_input = keras.Input(shape=(1,),dtype='int64')
    intercepto = layers.Embedding(1,1,embeddings_regularizer=reg_func(regpar_eff))(intercepto_input)
    intercepto_flatten = layers.Flatten()(intercepto)
    inputs.append(intercepto_input)
    layers_efeitos.append(intercepto_flatten)
    
    # Efeitos Numéricos
    layers_efeitos_num = []
    if len(var_eff_num) > 0:
        for variavel in var_eff_num:
            for i in np.arange(1,exp_p+1):
                var_i_name = variavel + '^' + str(i)
                locals()[var_i_name] = keras.Input(shape=(1,),dtype='float64')
                inputs.append(locals()[var_i_name])
                var_ii_name = variavel + '^' + str(i) + '_flatten'
                locals()[var_ii_name] = layers.Flatten()(locals()[var_i_name])
                layers_efeitos_num.append(locals()[var_ii_name])
        concat_efeitos_num = layers.Concatenate()(layers_efeitos_num)
        dense_efeitos_num = layers.Dense(1,activation='linear',kernel_regularizer=reg_func(regpar_eff))(concat_efeitos_num)
        layers_efeitos.append(dense_efeitos_num)
    
    # Efeitos Categóricos
    layers_efeitos_cat = []
    if len(var_eff_cat) > 0:
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
        layers_efeitos.append(dense_efeitos_cat)
    
    concat_efeitos = layers.Concatenate()(layers_efeitos)
    efeitos = layers.Dense(1,activation='linear',kernel_regularizer=reg_func(regpar_eff),kernel_initializer='ones',use_bias=False,trainable=False)(concat_efeitos)
    
    ### Espaço Latente
    
    if len(var_latU_num)+len(var_latU_cat) > 0 and len(var_latP_num)+len(var_latP_cat) > 0:
        # Espaço latente U (Votante)
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
        # Espaço latente P (Votacao)
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
        expoente_batch_size,
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
            add_epoch=100,
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
        performance[i] = BinaryCrossEntropy(
            y_true=np.array(y_true),
            y_pred=np.array(y_pred)
            )
    return -np.mean(performance)
