# Lista de Exercícios Parte 3 - Capítulo 11

# Obs: Caso tenha problemas com a acentuação, consulte este link:
# https://support.rstudio.com/hc/en-us/articles/200532197-Character-Encoding

# Configurando o diretório de trabalho
# Coloque entre aspas o diretório de trabalho que você está usando no seu computador
# Não use diretórios com espaço no nome
setwd("D:/Documentos/DataScienceAcademy/DSA_BigDataRAzure/Scripts/Cap11")
getwd()


# Definindo o Problema: Analisando dados das casas de Boston, nos EUA e fazendo previsoes.

# The Boston Housing Dataset
# http://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html

# Seu modelo deve prever a MEDV (Valor da Mediana de ocupação das casas). Utilize um modelo de rede neural!

# Carregando o pacote MASS
library(MASS)

# Importando os dados do dataset Boston
set.seed(101)
dados <- Boston
head(dados)

# Resumo dos dados
str(dados)
summary(dados)
any(is.na(dados))

# Carregando o pacote para Redes Neurais
# install.packages("neuralnet")
library(neuralnet)

# Análise Exploratória
quantile(dados$crim, seq( from = 0, to = 1, by = 0.10))

# Plot

# Boxplot
boxplot(dados$crim, main = "Boxplot para a Taxa de Crime per Capta por Cidade", ylab = "Taxa")


# Histograma
hist(dados$crim, main = "Histograma para a Taxa de Crime per Capta por Cidade", xlab = "Taxa")


# Vamos tentar desenvolver um modelo inicial básico para depois melhorá-lo
# Primeiramente, é necessário normalizar os dados, porque se n fazemos o modelo
# de redes neurais muitas vezes irá prever os mesmos resultados independente da entrada.

# Temos 2 possibilidades para normalizar: Scaled Normalization e Max-Min Normalization
# Inicialmente vou testar com a Max-Min Normalization

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

maxmindf <- as.data.frame(lapply(dados, normalize))

# Training and Test Data
sample <- sample(c(TRUE, FALSE), nrow(maxmindf), replace=TRUE, prob=c(0.7,0.3))
trainset  <- maxmindf[sample, ]
testset  <- maxmindf[!sample, ]

#Neural Network 1
nn <- neuralnet(medv ~ ., data=trainset, linear.output=FALSE, threshold=0.01)
nn$result.matrix
plot(nn)

#Test the resulting output
temp_test <- subset(testset, select = -medv)
head(temp_test)
nn.results <- compute(nn, temp_test)
results <- data.frame(actual = testset$medv, prediction = nn.results$net.result)

# Confusion Matrix
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)


#Neural Network 2 - Acrescentando o parâmetro 'hidden'
nn <- neuralnet(medv ~ ., data=trainset, hidden=c(2,1), linear.output=FALSE, threshold=0.01)
nn$result.matrix
plot(nn)

#Test the resulting output
temp_test <- subset(testset, select = -medv)
head(temp_test)
nn.results <- compute(nn, temp_test)
results <- data.frame(actual = testset$medv, prediction = nn.results$net.result)

# Confusion Matrix
roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
table(actual,prediction)

# Percebi aqui que estava utilizando o método de Classificação

# Refazendo para Regressão
# Importando os dados do dataset Boston
set.seed(101)
dados <- Boston
head(dados)

# Assim como na classificação foi feita a normalização e a separação em treino e teste

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

maxmindf <- as.data.frame(lapply(dados, normalize))

# Training and Test Data
sample <- sample(c(TRUE, FALSE), nrow(maxmindf), replace=TRUE, prob=c(0.7,0.3))
trainset  <- maxmindf[sample, ]
testset  <- maxmindf[!sample, ]

#4. NEURAL NETWORK
nn <- neuralnet(medv ~ .,data=trainset, hidden=c(2,1), linear.output=TRUE, threshold=0.01)
nn$result.matrix
plot(nn)


# Accuracy
#Test the resulting output
temp_test <- subset(testset, select = -c(medv))
head(temp_test)
nn.results <- compute(nn, temp_test)

# Model Validation
results <- data.frame(actual = testset$medv, prediction = nn.results$net.result)
results

# Convertendo dados para unidade inicial (desnormalizando)
predicted=results$prediction * abs(diff(range(dados$medv))) + min(dados$medv)
actual=results$actual * abs(diff(range(dados$medv))) + min(dados$medv)
head(predicted)
head(actual)

# Calculando a Acurácia
comparison=data.frame(predicted,actual)
deviation=((actual-predicted)/actual)
comparison=data.frame(predicted,actual,deviation)
accuracy=1-abs(mean(deviation))
accuracy

# O teste indica que com hidden=c(2,1) obtivemos uma acurácia de 93.48%




# Testando com outras configurações do parâmetro hidden para ver se conseguimos
# aumentar a performance.




# ################################  Regressão Testes  ########################################

# Importando os dados do dataset Boston
set.seed(101)
dados <- Boston
head(dados)

# Assim como na classificação foi feita a normalização e a separação em treino e teste

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

maxmindf <- as.data.frame(lapply(dados, normalize))

# Training and Test Data
library(caTools)
split = sample.split(maxmindf$medv, SplitRatio = 0.70)

trainset = subset(maxmindf, split == TRUE)
testset = subset(maxmindf, split == FALSE)

#4. NEURAL NETWORK
nn <- neuralnet(medv ~ .,data=trainset, hidden=c(7,7), linear.output=TRUE, threshold=0.01)
nn$result.matrix
plot(nn)

# Accuracy
# Test the resulting output
temp_test <- subset(testset, select = -c(medv))
head(temp_test)
nn.results <- compute(nn, temp_test)

# Model Validation
results <- data.frame(actual = testset$medv, prediction = nn.results$net.result)
results

# Convertendo dados para unidade inicial (desnormalizando)
predicted=results$prediction * abs(diff(range(dados$medv))) + min(dados$medv)
actual=results$actual * abs(diff(range(dados$medv))) + min(dados$medv)
head(predicted)
head(actual)

# Calculando a Acurácia
comparison=data.frame(predicted,actual)
deviation=((actual-predicted)/actual)
comparison=data.frame(predicted,actual,deviation)
accuracy=1-abs(mean(deviation))
accuracy

# O teste indica que com hidden=c(5,2) obtivemos uma acurácia de 94.32%
# O teste indica que com hidden=c(4,4) obtivemos uma acurácia de 94.83%
# O teste indica que com hidden=c(7,2) obtivemos uma acurácia de 94.83%
# O teste indica que com hidden=c(7,7) obtivemos uma acurácia de 99,59%

# Acredito que uma acurácia de 99,59% é mais que suficiente,



# Adicionando pós correção apresentada pelo Daniel

# Calculando o Mean Squared Error
MSE.nn <- sum((actual - predicted)^2)/nrow(testset)
MSE.nn

# Obtendo os erros de previsao
error.df <- data.frame(actual, predicted)
head(error.df)

# Plot dos erros
library(ggplot2)
ggplot(error.df, aes(x = actual,y = predicted)) + 
  geom_point() + stat_smooth()





























