#################
### Preâmbulo ###
#################

library(wnominate)
library(magrittr)
library(ggplot2)
library(ggtext)

diret = 'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4'

#########################
### Leitura dos dados ###
#########################

data(sen90)
AA = sen90$votes

######################################
### Exportação dos dados originais ###
######################################

nome_arquivo = file.path(diret,'0.Dados','D00. Original.csv')
AA %>%
  write.csv(file=nome_arquivo)

############################################
### Exportação dos dados sem votações NA ###
############################################

# Rodando modelo W-NOMINATE segundo o próprio pacote
sen90wnom = wnominate(sen90,polarity=c(2,5))
# Filtrando apenas votações consideradas pelo W-NOMINATE
BB = AA[,(!is.na(sen90wnom$rollcalls[,c(1)]))]
# Exportando arquivo
nome_arquivo = file.path(diret,'0.Dados','D00. Matriz sem votações NA.csv')
BB %>%
  write.csv(file=nome_arquivo)

############################################################
### Aplicação do W-NOMINATE variando número de dimensões ###
############################################################

# Aplicação com 1 dimensão
mod = wnominate(sen90,dims=1,polarity=c(1))
# Cálculo da acurácia removendo NA
sum(mod$rollcalls[,c(1,4)],na.rm=TRUE)/sum(mod$rollcalls[,c(1:4)],na.rm=TRUE) # 0.8047158

# Aplicação com 2 dimensões
mod = wnominate(sen90,dims=2,polarity=c(1,2))
# Cálculo da acurácia removendo NA
sum(mod$rollcalls[,c(1,4)],na.rm=TRUE)/sum(mod$rollcalls[,c(1:4)],na.rm=TRUE) # 0.8426787

# Aplicação com 3 dimensões
mod = wnominate(sen90,dims=3,polarity=c(1,2,3))
# Cálculo da acurácia removendo NA
sum(mod$rollcalls[,c(1,4)],na.rm=TRUE)/sum(mod$rollcalls[,c(1:4)],na.rm=TRUE) # 0.8590224

####################################
### Escolha do Modelo e Plotagem ###
####################################

setwd(diret)

### DIM 1

mod = wnominate(sen90,dims=1,polarity=c(1))

df = data.frame(
  "Numero" = 1:102,
  "Votante" = rownames(mod$legislators),
  "Partido" = mod$legislators$party,
  "Coord1D" = mod$legislators$coord1D,
  "Constante" = rep(0,102)
)
df[!(df$Numero %in% c(17,75,3,34)),"Votante"] = ""

pdf("8.Extra/D00_Graficos_WNOM_DIM1.pdf",width=8,height=4)
df %>%
  ggplot(aes(x=Coord1D,y=Constante,col=Partido,label=Votante))+
  geom_point(size=3,alpha=0.3)+
  geom_text(angle=90,show.legend=FALSE,col="black",vjust="inward",hjust="inward")+
  theme_minimal()+
  xlab("Coordenada 1")+
  ylab("Constante")
df %>%
  ggplot(aes(x=Coord1D,fill=Partido))+
  geom_density(alpha=0.5)+
  theme_minimal()+
  xlab("Coordenada 1")+
  ylab("Densidade")
dev.off()

### DIM 2

mod = wnominate(sen90,dims=2,polarity=c(1,2))

df = data.frame(
  "Numero" = 1:102,
  "Votante" = rownames(mod$legislators),
  "Partido" = mod$legislators$party,
  "Coord1D" = mod$legislators$coord1D,
  "Coord2D" = mod$legislators$coord2D
)
df[!(df$Numero %in% c(17,75,3,34)),"Votante"] = ""

pdf("8.Extra/D00_Graficos_WNOM_DIM2.pdf",width=8,height=4)
df %>%
  ggplot(aes(x=Coord1D,y=Coord2D,col=Partido,label=Votante))+
  geom_point(size=3,alpha=0.3)+
  geom_text(angle=90,show.legend=FALSE,col="black",vjust="inward",hjust="inward")+
  theme_minimal()+
  xlab("Coordenada 1")+
  ylab("Coordenada 2")
df %>%
  ggplot(aes(x=Coord1D,fill=Partido))+
  geom_density(alpha=0.5)+
  theme_minimal()+
  xlab("Coordenada 1")+
  ylab("Densidade")
df %>%
  ggplot(aes(x=Coord2D,fill=Partido))+
  geom_density(alpha=0.5)+
  theme_minimal()+
  xlab("Coordenada 2")+
  ylab("Densidade")
dev.off()

### DIM 2 SEL

mod = wnominate(sen90,dims=2,polarity=c(2,5))

df = data.frame(
  "Numero" = 1:102,
  "Votante" = rownames(mod$legislators),
  "Partido" = mod$legislators$party,
  "Coord1D" = mod$legislators$coord1D,
  "Coord2D" = mod$legislators$coord2D
)
df[!(df$Numero %in% c(17,75,3,34)),"Votante"] = ""

pdf("8.Extra/D00_Graficos_WNOM_DIM2_SEL.pdf",width=8,height=4)
df %>%
  ggplot(aes(x=Coord1D,y=Coord2D,col=Partido,label=Votante))+
  geom_point(size=3,alpha=0.3)+
  geom_text(angle=90,show.legend=FALSE,col="black",vjust="inward",hjust="inward")+
  theme_minimal()+
  xlab("Coordenada 1")+
  ylab("Coordenada 2")
df %>%
  ggplot(aes(x=Coord1D,fill=Partido))+
  geom_density(alpha=0.5)+
  theme_minimal()+
  xlab("Coordenada 1")+
  ylab("Densidade")
df %>%
  ggplot(aes(x=Coord2D,fill=Partido))+
  geom_density(alpha=0.5)+
  theme_minimal()+
  xlab("Coordenada 2")+
  ylab("Densidade")
dev.off()
