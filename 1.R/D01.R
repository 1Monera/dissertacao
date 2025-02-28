#################
### Preâmbulo ###
#################

library(magrittr)
library(dplyr)
library(data.table)
library(wnominate)
library(ggplot2)
library(ggtext)

diret = 'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V3'

#########################
### Leitura dos dados ###
#########################

caminho = file.path(diret,'0.Dados','D01. ADINs - Jeferson Mariano Jurisdicao_constitucional_no_Brasil_1966.csv')
AA = read.csv(caminho,sep=',')
BB = AA %>%
  as.data.table() %>%
  dcast(id_votante~id_votacao,value.var="voto") %>%
  as.data.frame()
rownames(BB) = BB$id_votante
CC = BB %>%
  select(-id_votante)
RC = rollcall(CC,yea=1,nay=0,missing=NA)

############################################################
### Aplicação do W-NOMINATE variando número de dimensões ###
############################################################

# Aplicação com 1 dimensão
mod = wnominate(RC,dims=1,polarity=c(2))
# Cálculo da acurácia removendo NA
sum(mod$rollcalls[,c(1,4)],na.rm=TRUE)/sum(mod$rollcalls[,c(1:4)],na.rm=TRUE) # 0.8472862

# Aplicação com 2 dimensões
mod = wnominate(RC,dims=2,polarity=c(2,3))
# Cálculo da acurácia removendo NA
sum(mod$rollcalls[,c(1,4)],na.rm=TRUE)/sum(mod$rollcalls[,c(1:4)],na.rm=TRUE) # 0.8990335

# Aplicação com 3 dimensões
mod = wnominate(RC,dims=3,polarity=c(2,3,5))
# Cálculo da acurácia removendo NA
sum(mod$rollcalls[,c(1,4)],na.rm=TRUE)/sum(mod$rollcalls[,c(1:4)],na.rm=TRUE) # 0.9112268

############################################
### Exportação dos dados sem votações NA ###
############################################

# Filtrando apenas votações consideradas pelo W-NOMINATE
DD = CC[(!is.na(mod$legislators[,c(1)])),]
# Exportando arquivo
nome_arquivo = file.path(diret,'0.Dados','D01. ADINs sem votações NA.csv')
DD %>%
  write.csv(file=nome_arquivo)

####################################
### Escolha do Modelo e Plotagem ###
####################################
mod = wnominate(RC,dims=2,polarity=c(2,3))

pos_votante_wnom = (1:32)[!is.na(mod$legislators[,1])]
pos_votacao_wnom = c(1:704)[!is.na(mod$rollcalls[,1])]

df = data.frame(
  "Numero" = 1:30,
  "Votante" = rownames(mod$legislators)[pos_votante_wnom],
  "Coord1D" = mod$legislators$coord1D[pos_votante_wnom],
  "Coord2D" = mod$legislators$coord2D[pos_votante_wnom]
)
depara = file.path(diret,'3.Auxiliares','D01. AB DE_PARA.csv') %>%
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
pdf("8.Extra/D01_Graficos_WNOM.pdf",width=5,height=5)
df %>%
  ggplot(aes(x=Coord1D,y=Coord2D,label=Nome))+
  geom_point(size=3,col="black",fill="black",alpha=0.5)+
  geom_text(angle=0,col="black",check_overlap=TRUE,vjust="inward",hjust="inward")+
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
