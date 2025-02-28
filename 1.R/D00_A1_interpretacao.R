#################
### Preâmbulo ###
#################

library(wnominate)
library(magrittr)
library(ggplot2)
library(ggtext)

diret = 'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4'

# Votante
caminho = file.path(diret,'3.Auxiliares','D00_A1_Votante.csv')
df = read.csv(caminho,sep=',')
df[!(df$id %in% c(15,1,32,72)),"votante"] = ""
df$Jitter = 0

setwd(diret)
pdf("8.Extra/D00_Graficos_FM-COV_Votante.pdf",width=8,height=4)
df %>%
  ggplot(aes(x=alpha,y=Jitter,col=partido,label=votante))+
  geom_point(size=3,alpha=0.5)+
  geom_text(angle=90,show.legend=FALSE,col="black")+
  theme_minimal()+
  xlab("Alpha Votante")+
  ylab("Constante")
df %>%
  ggplot(aes(x=dim_latente,y=Jitter,col=partido,label=votante))+
  geom_point(size=3,alpha=0.3)+
  geom_text(angle=90,show.legend=FALSE,col="black")+
  theme_minimal()+
  xlab("DimLat1 Votante")+
  ylab("Constante")
dev.off()

# Votacao
caminho = file.path(diret,'3.Auxiliares','D00_A1_Votacao.csv')
df = read.csv(caminho,sep=',')
df[!(df$id %in% c(0,10,20)),"votacao"] = ""
df$Jitter = 0

setwd(diret)
pdf("8.Extra/D00_Graficos_FM-COV_Votacao.pdf",width=8,height=4)
df %>%
  ggplot(aes(x=alpha,y=Jitter,label=votacao))+
  geom_point(size=3,alpha=0.3)+
  geom_text(angle=90,show.legend=FALSE,col="red")+
  theme_minimal()+
  xlab("Alpha Votacao")+
  ylab("Constante")
df %>%
  ggplot(aes(x=dim_latente,y=Jitter,label=votacao))+
  geom_point(size=3,alpha=0.3)+
  geom_text(angle=90,show.legend=FALSE,col="red")+
  theme_minimal()+
  xlab("DimLat1 Votacao")+
  ylab("Constante")
dev.off()
