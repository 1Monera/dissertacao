# Bibliotecas
import os
import numpy as np
import pandas as pd

diretorio = r'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V3'

# Leitura dos dados dos votos
anos = np.linspace(2001,2023,num=2023-2001+1)
for ano in anos:
    # Lê os dados
    url = 'http://dadosabertos.camara.leg.br/arquivos/votacoesVotos/csv/votacoesVotos-'+str(int(ano))+'.csv'
    dt_corrente = pd.read_csv(url,sep=';',usecols=['idVotacao','dataHoraVoto','voto','deputado_id','deputado_nome','deputado_siglaPartido','deputado_siglaUf'])
    dt_corrente = dt_corrente.loc[~pd.isnull(dt_corrente.voto)]
    # Constrói uma base única
    if ano == 2001:
        dt_full = dt_corrente
    else:
        dt_full = pd.concat([dt_full,dt_corrente])
dt_votos = dt_full.reset_index(drop=True)

# Leitura dos dados das votações
anos = np.linspace(2001,2023,num=2023-2001+1)
for ano in anos:
    # Lê os dados
    url = 'http://dadosabertos.camara.leg.br/arquivos/votacoes/csv/votacoes-'+str(int(ano))+'.csv'
    dt_corrente = pd.read_csv(url,sep=';',usecols=['id','siglaOrgao','aprovacao','votosSim','votosNao','votosOutros'])
    # Constrói uma base única
    if ano == 2001:
        dt_full = dt_corrente
    else:
        dt_full = pd.concat([dt_full,dt_corrente])
dt_votacoes = dt_full.reset_index(drop=True).rename(columns={"id": "idVotacao"})

dt = dt_votos.merge(dt_votacoes,on='idVotacao',how='left')
dt = dt\
    .loc[dt.siglaOrgao == 'PLEN']\
    .loc[dt.aprovacao.isin([0,1])]\
    .loc[dt.voto.isin(['Sim','Não'])]\
    .loc[~(dt.votosSim == 0)]\
    .loc[~(dt.votosNao == 0)]\
    .reset_index(drop=True)

# Verificar que as votações remanescentes não são unânimes
teste = pd.pivot_table(dt,index=['idVotacao'],columns=['voto'],aggfunc="count",values='aprovacao')
teste.loc[teste['Sim']==0] # OK
teste.loc[teste['Não']==0] # OK

# Verificar inconsistências de duplicação de deputado_id
teste = dt[['deputado_id','deputado_nome']]\
            .drop_duplicates()\
            .reset_index(drop=True)
teste1 = teste.groupby(['deputado_id']).count()
ids_dup = teste1.loc[teste1.deputado_nome > 1].index
caminho = os.path.join(diretorio,'3.Auxiliares','deputados_id_nome.csv')
teste.loc[teste.deputado_id.isin(ids_dup)].sort_values(['deputado_id']).to_csv(caminho,index=False,sep=';',encoding='latin-1')
# Padroniza nome do deputado
caminho = os.path.join(diretorio,'3.Auxiliares','deputados_id_nome_refeito.csv')
aux = pd.read_csv(caminho,encoding='latin-1',sep=';')
for i in range(aux.shape[0]):
    corrente_id = aux.loc[i].deputado_id
    corrente_nome = aux.loc[i].deputado_nome
    dt['deputado_nome'] = np.where(dt['deputado_id']==corrente_id,corrente_nome,dt['deputado_nome'])

# Verificar inconsistências de duplicação de deputado_nome
teste = dt[['deputado_id','deputado_nome']]\
            .drop_duplicates()\
            .reset_index(drop=True)
teste1 = teste.groupby(['deputado_nome']).count()
nomes_dup = teste1.loc[teste1.deputado_id > 1].index
caminho = os.path.join(diretorio,'3.Auxiliares','deputados_nome_id.csv')
teste.loc[teste.deputado_nome.isin(nomes_dup)].sort_values(['deputado_nome']).to_csv(caminho,index=False,sep=';',encoding='latin-1')
# Padroniza nome do deputado
caminho = os.path.join(diretorio,'3.Auxiliares','deputados_nome_id_refeito.csv')
aux = pd.read_csv(caminho,encoding='latin-1',sep=';')
for i in range(aux.shape[0]):
    corrente_id = aux.loc[i].deputado_id
    corrente_nome = aux.loc[i].deputado_nome
    dt['deputado_nome'] = np.where(dt['deputado_id']==corrente_id,corrente_nome,dt['deputado_nome'])

# Extrai data
dt['Data'] = pd.to_datetime(dt.dataHoraVoto,format='%Y-%m-%dT%H:%M:%S').dt.date

# Salva D03
caminho = os.path.join(diretorio,'0.Dados','D03. Camara Deputados PLEN.csv')
dt\
    [['Data','deputado_nome','deputado_siglaPartido','deputado_siglaUf','idVotacao','voto']]\
    .to_csv(caminho,index=False,encoding='latin-1')
