import os
import pandas as pd
import numpy as np

#########################
# Codificação dos votos #
#########################
def f_padroniza_dados(df):
    # Selecionando colunas de interesse
    info_votos = df[['ID_STF',
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
    
    info_votos = info_votos.rename(columns={'ID_STF': 'Votacao'})
    
    # Alterando formato da base de dados
    info_votos = pd.melt(info_votos,id_vars='Votacao',var_name='Votante',value_name='voto')
    
    # Removendo valores (votos) faltantes
    info_votos = info_votos.dropna().reset_index(drop=True)
    
    # Substituindo votos por valores binários
    for linha in range(info_votos.shape[0]):
        voto = info_votos.loc[linha,'voto']
        if voto == 'Vencedor(a)':
            # Votante concorda com a maioria
            voto_pad = 1
        elif voto == 'DERROTADO(A)':
            # Votante discorda da maioria
            voto_pad = 0
        info_votos.loc[linha,'voto'] = voto_pad
    
    return info_votos

######################
# Serializa votantes #
######################
def f_serializa_votantes(info_votos):
    # Lista todos os Votantes
    de_para = pd.DataFrame({'Votante':info_votos.Votante.unique().tolist()})
    # Atribui uma numeração sequencial para cada Votante
    de_para['id_votante'] = np.arange(0,de_para.shape[0])
    # Substitui nome do Votante pela numeração serial
    info_votos = info_votos.merge(de_para,how='left',left_on='Votante',right_on='Votante').drop(['Votante'],axis=1)
    return info_votos

######################
# Serializa votacoes #
######################
def f_serializa_votacoes(info_votos):
    # Lista todos as Votações
    de_para = pd.DataFrame({'Votacao':info_votos.Votacao.unique().tolist()})
    # Atribui uma numeração sequencial para cada Votação
    de_para['id_votacao'] = np.arange(0,de_para.shape[0])
    # Substitui nome do Votação pela numeração serial
    info_votos = info_votos.merge(de_para,how='left',left_on='Votacao',right_on='Votacao').drop(['Votacao'],axis=1)
    return info_votos

# Salva diretório
diretorio = r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V3'

# Lê dados
caminho = os.path.join(diretorio,'0.Dados','D01. ADINs - Jeferson Mariano Jurisdicao_constitucional_no_Brasil_1966.xlsx')
df = pd.read_excel(caminho,sheet_name='ADIns')
# Filtra Dados
df = df[df['Julgamento_resultado'] != 'Aguardando']
df = df[df['Julgamento_votação'] == 'Maioria']
df = df.reset_index(drop=True)
# Padroniza de acordo com conceito
df = f_padroniza_dados(df)
df = f_serializa_votantes(df)
df = f_serializa_votacoes(df)
# Salva em CSV para outros propósitos
diretorio = r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V3'
caminho = os.path.join(diretorio,'0.Dados','D01. ADINs - Jeferson Mariano Jurisdicao_constitucional_no_Brasil_1966.csv')
df.to_csv(caminho,index=False)
