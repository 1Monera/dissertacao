#################
### Preâmbulo ###
#################

library(wnominate)
library(magrittr)
library(ggplot2)
library(ggtext)

diret = 'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4'

# Votante
caminho = file.path(diret,'3.Auxiliares','D02_A0_Votante.csv')
df = read.csv(caminho,sep=',')
#df[abs(df$alpha) <= 0.01,"votante"] = ""
df$Jitter = 0

setwd(diret)
pdf("8.Extra/D02_Graficos_FM-COV_Votante.pdf",width=8,height=4)
df %>%
  ggplot(aes(x=alpha,y=Jitter,label=votante))+
  geom_point(size=3,alpha=0.3)+
  geom_text(angle=90,check_overlap=TRUE,show.legend=FALSE,col="red")+
  theme_minimal()+
  xlab("Alpha Votante")+
  ylab("Constante")
dev.off()

# Votacao
caminho = file.path(diret,'3.Auxiliares','D02_A0_Votacao.csv')
df = read.csv(caminho,sep=',')
#df[!(df$id %in% c(168)),"votacao"] = ""
df$Jitter = 0

setwd(diret)
pdf("8.Extra/D02_Graficos_FM-COV_Votacao.pdf",width=8,height=4)
df %>%
  ggplot(aes(x=alpha,y=Jitter,label=votacao))+
  geom_point(size=3,alpha=0.3)+
  geom_text(angle=90,check_overlap=TRUE,show.legend=FALSE,col="red")+
  theme_minimal()+
  xlab("Alpha Votacao")+
  ylab("Constante")
dev.off()
