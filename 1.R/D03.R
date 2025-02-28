#################
### Preâmbulo ###
#################

library(magrittr)
library(dplyr)
library(data.table)
library(wnominate)

diret = 'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V3'

#########################
### Leitura dos dados ###
#########################

caminho = file.path(diret,'0.Dados','D03. WNOMINATE-FORMAT.csv')
AA = read.csv(caminho,sep=',')
BB = AA %>%
  as.data.table() %>%
  dcast(Votante~Votacao,value.var="Voto") %>%
  as.data.frame()
rownames(BB) = BB$Votante
CC = BB %>%
  select(-Votante)
RC = rollcall(CC,yea=1,nay=0,missing=NA)

############################################################
### Aplicação do W-NOMINATE variando número de dimensões ###
############################################################

mod = wnominate(RC,dims=1,polarity=c(1))
votante_pos = (1:length(rownames(mod$legislators)))[is.na(mod$legislators[,1])]
votacao_pos = (1:length(rownames(mod$rollcalls)))[is.na(mod$rollcalls[,1])]
votante_drop = rownames(CC)[votante_pos] # Votantes Dropados pelo W-NOMINATE
votacao_drop = colnames(CC)[votacao_pos] # Votações Dropadas pelo W-NOMINATE

# Aplicação com 1 dimensão
mod = wnominate(RC,dims=1,polarity=c(1))
# Cálculo da acurácia removendo NA
sum(mod$rollcalls[,c(1,4)],na.rm=TRUE)/sum(mod$rollcalls[,c(1:4)],na.rm=TRUE) # 0.8864525

# Aplicação com 2 dimensões
mod = wnominate(RC,dims=2,polarity=c(1,2))
# Cálculo da acurácia removendo NA
sum(mod$rollcalls[,c(1,4)],na.rm=TRUE)/sum(mod$rollcalls[,c(1:4)],na.rm=TRUE) # 0.9000501

# Aplicação com 3 dimensões
mod = wnominate(RC,dims=3,polarity=c(1,2,3))
# Cálculo da acurácia removendo NA
sum(mod$rollcalls[,c(1,4)],na.rm=TRUE)/sum(mod$rollcalls[,c(1:4)],na.rm=TRUE) # 0.9062978

############################################
### Exportação dos dados sem votações NA ###
############################################

# Filtrando apenas votações consideradas pelo W-NOMINATE
DD = AA %>%
  filter(!Votante %in% votante_drop) %>%
  filter(!Votacao %in% votacao_drop)
# Exportando arquivo
nome_arquivo = file.path(diret,'0.Dados','D03. CamaraDeputados reduzido WNOM.csv')
DD %>%
  write.csv(file=nome_arquivo)

####################################
### Escolha do Modelo e Plotagem ###
####################################
#mod = wnominate(RC,dims=2,polarity=c(1,2)) # 7000 segundos (1h12) para rodar
#saveRDS(mod,file=file.path(diret,"3.Auxiliares","D03_Modelo_2DIM.RDS"))
mod = readRDS(file.path(diret,"3.Auxiliares","D03_Modelo_2DIM.RDS"))

pos_votante_wnom = (1:2141)[!is.na(mod$legislators[,1])]

df = data.frame(
  "Votante" = rownames(mod$legislators)[pos_votante_wnom],
  "Coord1D" = mod$legislators$coord1D[pos_votante_wnom],
  "Coord2D" = mod$legislators$coord2D[pos_votante_wnom]
)
depara = file.path(diret,'3.Auxiliares','D03. AB DE_PARA.csv') %>%
  read.csv(sep=",") %>%
  dplyr::filter(Variavel == "Votante") %>%
  mutate(
    ID = ID+1,
    Votante = paste0("Legislator ",ID),
    Nome = gsub('2','',Descricao)
  ) %>%
  select(Votante,Nome)
df = merge(x=df,y=depara,by="Votante",all.x=TRUE) %>%
  na.omit()

setwd(diret)
pdf("8.Extra/D03_Graficos_WNOM.pdf",width=5,height=5)
df %>%
  ggplot(aes(x=Coord1D,y=Coord2D,label=Nome))+
  geom_point(size=3,col="black",fill="black",alpha=0.5)+
  geom_text(angle=0,col="blue",check_overlap=TRUE,vjust="inward",hjust="inward")+
  theme_minimal()+
  ggtitle("Plotagem das Dimensões Latentes")+
  xlab("Coordenada 1")+
  ylab("Coordenada 2")
# df %>%
#   ggplot(aes(x=Coord1D))+
#   geom_density(alpha=0.5,fill="black")+
#   theme_minimal()+
#   coord_cartesian(xlim=c(-1,1))+
#   ggtitle("Densidade da Dimensão Latente")+
#   xlab("Coordenada 1")+
#   ylab("Densidade")
# df %>%
#   ggplot(aes(x=Coord2D))+
#   geom_density(alpha=0.5,fill="black")+
#   theme_minimal()+
#   coord_cartesian(xlim=c(-1,1))+
#   ggtitle("Densidade da Dimensão Latente")+
#   xlab("Coordenada 2")+
#   ylab("Densidade")
dev.off()
