#################
### Preâmbulo ###
#################

library(wnominate)
library(magrittr)
library(ggplot2)
library(ggtext)

diret = 'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4'
setwd(diret)

for(i in 1:2){
  
  ### Cria PDF
  if(i == 1){
    pdf("8.Extra/D03_Graficos_FM-COV_Alpha.pdf",width=8,height=4)
  }else if(i == 2){
    pdf("8.Extra/D03_Graficos_FM-COV_DimLat.pdf",width=5,height=5)
  }
  
  ### Votante
  caminho = file.path(diret,'3.Auxiliares','D03_B2_Votante.csv')
  df = read.csv(caminho,sep=',')
  df$Jitter = 0
  if(i == 1){
    g1 = df %>%
      ggplot(aes(x=alpha,y=Jitter,label=votante))+
      geom_point(size=3,alpha=0.3)+
      geom_text(angle=90,check_overlap=TRUE,show.legend=FALSE,col="red")+
      theme_minimal()+
      xlab("Alpha Votante")+
      ylab("Constante")
  }else if(i == 2){
    g1 = df %>%
      ggplot(aes(x=dim_lat1,y=dim_lat2,label=votante))+
      geom_point(size=3,alpha=0.3)+
      geom_text(check_overlap=TRUE,show.legend=FALSE,col="red",vjust="inward",hjust="inward")+
      theme_minimal()+
      xlab("Dimensão Latente 1")+
      ylab("Dimensão Latente 2")
  }
  print(g1)
  
  ### Partido
  caminho = file.path(diret,'3.Auxiliares','D03_B2_Partido.csv')
  df = read.csv(caminho,sep=',')
  df$Jitter = 0
  if(i == 1){
    g2 = df %>%
      ggplot(aes(x=alpha,y=Jitter,label=partido))+
      geom_point(size=3,alpha=0.3)+
      geom_text(angle=90,check_overlap=TRUE,show.legend=FALSE,col="red")+
      theme_minimal()+
      xlab("Alpha Partido")+
      ylab("Constante")
  }else if(i == 2){
    g2 = df %>%
      ggplot(aes(x=dim_lat1,y=dim_lat2,label=partido))+
      geom_point(size=3,alpha=0.3)+
      geom_text(check_overlap=TRUE,show.legend=FALSE,col="red")+
      theme_minimal()+
      xlab("Dimensão Latente 1")+
      ylab("Dimensão Latente 2")
  }
  print(g2)
  
  ### UF
  caminho = file.path(diret,'3.Auxiliares','D03_B2_UF.csv')
  df = read.csv(caminho,sep=',')
  df$Jitter = 0
  if(i == 1){
    g3 = df %>%
      ggplot(aes(x=alpha,y=Jitter,label=uf))+
      geom_point(size=3,alpha=0.3)+
      geom_text(angle=90,check_overlap=TRUE,show.legend=FALSE,col="red")+
      theme_minimal()+
      xlab("Alpha UF")+
      ylab("Constante")
  }else if(i == 2){
    g3 = df %>%
      ggplot(aes(x=dim_lat1,y=dim_lat2,label=uf))+
      geom_point(size=3,alpha=0.3)+
      geom_text(check_overlap=TRUE,show.legend=FALSE,col="red")+
      theme_minimal()+
      xlab("Dimensão Latente 1")+
      ylab("Dimensão Latente 2")
  }
  print(g3)
  
  ### Votacao
  caminho = file.path(diret,'3.Auxiliares','D03_B2_Votacao.csv')
  df = read.csv(caminho,sep=',')
  df$Jitter = 0
  if(i == 1){
    g4 = df %>%
      ggplot(aes(x=alpha,y=Jitter,label=votacao))+
      geom_point(size=3,alpha=0.3)+
      geom_text(angle=90,check_overlap=TRUE,show.legend=FALSE,col="red")+
      theme_minimal()+
      xlab("Alpha Votacao")+
      ylab("Constante")
  }else if(i == 2){
    g4 = df %>%
      ggplot(aes(x=dim_lat1,y=dim_lat2,label=votacao))+
      geom_point(size=3,alpha=0.3)+
      geom_text(check_overlap=TRUE,show.legend=FALSE,col="red")+
      theme_minimal()+
      xlab("Dimensão Latente 1")+
      ylab("Dimensão Latente 2")
  }
  print(g4)
  
  ### Fecha PDF
  dev.off()
  print(i)
}