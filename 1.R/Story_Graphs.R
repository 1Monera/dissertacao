library(magrittr)
library(dplyr)
library(ggplot2)

diret = 'C:/Users/Maurício/Google Drive/Estudo/Mestrado/Pesquisas/Dissertação/Códigos V4'
setwd(diret)

df = data.frame(
  "dados" = rep(c("D00","D01","D02","D03"),each=12),
  "cov" = rep(c("SCOV","CCOV"),24),
  "filtro" = rep(c("ALL","ALL","WNOM","WNOM"),12),
  "dim" = rep(rep(c(0,1,2),each=4),4),
  "auc" = c(81.3,81.3,75.5,75.6,89.6,89.4,85.4,85.8,86.7,91.6,86.5,85.3,
            72.1,71.6,67.1,69.0,74.9,73.6,72.1,71.7,74.7,73.7,70.3,72.6,
            72.6,69.2,68.4,60.7,73.1,69.7,64.8,59.4,72.6,67.4,68.4,56.8,
            86.7,83.3,83.1,78.1,90.5,91.0,85.2,90.4,95.4,95.8,94.0,94.8),
  "acc" = c(72.5,72.4,68.0,68.2,80.5,80.4,76.5,76.5,77.8,82.8,77.1,75.7,
            82.3,82.3,81.3,81.5,82.7,82.1,82.3,82.3,83.0,82.4,81.8,82.6,
            84.1,84.1,83.6,82.1,84.1,84.1,82.1,80.6,84.1,84.1,83.6,80.6,
            80.6,77.6,77.5,72.5,80.9,81.3,78.0,80.7,87.4,88.1,85.3,86.4),
  "f1" = c(76.7,76.7,68.4,69.0,83.3,83.1,76.9,78.4,80.9,85.0,77.6,75.2,
           89.6,89.6,89.0,89.1,89.9,89.5,89.6,89.7,90.1,89.7,89.5,89.6,
           91.4,91.4,90.8,89.7,91.4,91.4,90.0,89.3,91.4,91.4,90.8,89.3,
           82.8,78.6,79.7,73.6,83.1,83.2,80.3,81.8,88.4,89.0,85.9,87.0)
)

# Transformação
df = df %>%
  mutate(
    auc = auc/100,
    acc = acc/100,
    f1 = f1/100
  )

##### D00

# Gráfico 1
#width=10,height=9
pdf("8.Extra/D00_Perf_FM-COV_1.pdf",width=7,height=6)
df %>%
  filter(dados == "D00") %>%
  select(-dados) %>%
  data.table::as.data.table() %>%
  data.table::melt.data.table(id.vars=c("cov","filtro","dim")) %>%
  as.data.frame() %>%
  mutate(variable = ifelse(variable == "auc","AUC",ifelse(variable == "acc","Acurácia","F1"))) %>%
  ggplot(aes(x=dim,y=value,col=filtro,group=dim))+
  geom_point()+
  geom_line(col="black",size=0.5)+
  facet_grid(rows=vars(cov),cols=vars(variable))+
  xlab("# Dimensões Latentes")+
  ylab("Métrica")+
  scale_color_manual(values=c("ALL"="#00BFC4","WNOM"="#F8766D"))+
  scale_x_continuous(breaks = 0:2)+
  labs(color='Filtro')+
  theme_bw(base_size=12)
dev.off()
# Gráfico 2
pdf("8.Extra/D00_Perf_FM-COV_2.pdf",width=7,height=3)
df %>%
  filter(dados == "D00" & filtro == "ALL") %>%
  select(-c(dados,filtro)) %>%
  data.table::as.data.table() %>%
  data.table::melt.data.table(id.vars=c("cov","dim")) %>%
  as.data.frame() %>%
  mutate(variable = ifelse(variable == "auc","AUC",ifelse(variable == "acc","Acurácia","F1"))) %>%
  ggplot(aes(x=dim,y=value,col=cov,group=dim))+
  geom_point()+
  geom_line(col="black",size=0.5)+
  facet_grid(cols=vars(variable))+
  xlab("# Dimensões Latentes")+
  ylab("Métrica")+
  scale_color_manual(values=c("SCOV"="#00BFC4","CCOV"="#F8766D"))+
  scale_x_continuous(breaks = 0:2)+
  labs(color='Covariáveis')+
  theme_bw(base_size=12)
dev.off()

##### D03

# Gráfico 1
pdf("8.Extra/D03_Perf_FM-COV_1.pdf",width=7,height=3)
df %>%
  filter(dados == "D03" & filtro == "ALL") %>%
  select(-c(dados,filtro)) %>%
  data.table::as.data.table() %>%
  data.table::melt.data.table(id.vars=c("cov","dim")) %>%
  as.data.frame() %>%
  mutate(variable = ifelse(variable == "auc","AUC",ifelse(variable == "acc","Acurácia","F1"))) %>%
  ggplot(aes(x=dim,y=value,col=cov,group=dim))+
  geom_point()+
  geom_line(col="black",size=0.5)+
  facet_grid(cols=vars(variable))+
  xlab("# Dimensões Latentes")+
  ylab("Métrica")+
  scale_color_manual(values=c("SCOV"="#F8766D","CCOV"="#00BFC4"))+
  scale_x_continuous(breaks = 0:2)+
  labs(color='Covariáveis')+
  theme_bw(base_size=12)
dev.off()
# Gráfico 2
pdf("8.Extra/D03_Perf_FM-COV_2.pdf",width=7,height=3)
df %>%
  filter(dados == "D03" & filtro == "ALL" & cov == "CCOV") %>%
  select(-c(dados,filtro,cov)) %>%
  data.table::as.data.table() %>%
  data.table::melt.data.table(id.vars=c("dim")) %>%
  as.data.frame() %>%
  mutate(variable = ifelse(variable == "auc","AUC",ifelse(variable == "acc","Acurácia","F1"))) %>%
  ggplot(aes(x=dim,y=value))+
  geom_point()+
  geom_line(col="black",size=0.5)+
  facet_grid(cols=vars(variable))+
  xlab("# Dimensões Latentes")+
  ylab("Métrica")+
  scale_x_continuous(breaks = 0:2)+
  theme_bw(base_size=12)
dev.off()

##### D01

# Gráfico 1
pdf("8.Extra/D01_Perf_FM-COV_1.pdf",width=7,height=3)
df %>%
  filter(dados == "D01" & filtro == "ALL") %>%
  select(-dados,-filtro) %>%
  data.table::as.data.table() %>%
  data.table::melt.data.table(id.vars=c("cov","dim")) %>%
  as.data.frame() %>%
  mutate(variable = ifelse(variable == "auc","AUC",ifelse(variable == "acc","Acurácia","F1"))) %>%
  ggplot(aes(x=dim,y=value,col=cov,group=dim))+
  geom_point()+
  geom_line(col="black",size=0.5)+
  facet_grid(cols=vars(variable))+
  xlab("# Dimensões Latentes")+
  ylab("Métrica")+
  scale_color_manual(values=c("SCOV"="#00BFC4","CCOV"="#F8766D"))+
  scale_x_continuous(breaks = 0:2)+
  labs(color='Covariáveis')+
  theme_bw(base_size=12)
dev.off()

##### D02

# Gráfico 1
pdf("8.Extra/D02_Perf_FM-COV_1.pdf",width=7,height=3)
df %>%
  filter(dados == "D02" & filtro == "ALL") %>%
  select(-dados,-filtro) %>%
  data.table::as.data.table() %>%
  data.table::melt.data.table(id.vars=c("cov","dim")) %>%
  as.data.frame() %>%
  mutate(variable = ifelse(variable == "auc","AUC",ifelse(variable == "acc","Acurácia","F1"))) %>%
  ggplot(aes(x=dim,y=value,col=cov,group=dim))+
  geom_point()+
  geom_line(col="black",size=0.5)+
  facet_grid(cols=vars(variable))+
  xlab("# Dimensões Latentes")+
  ylab("Métrica")+
  scale_color_manual(values=c("SCOV"="#00BFC4","CCOV"="#F8766D"))+
  scale_x_continuous(breaks = 0:2)+
  labs(color='Covariáveis')+
  theme_bw(base_size=12)
dev.off()
