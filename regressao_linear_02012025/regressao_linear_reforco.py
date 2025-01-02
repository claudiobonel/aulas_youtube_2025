import pandas as pd
import numpy as np

endereco = r'C:\Users\claud\OneDrive\Claudio Bonel-DADOTECA\_ProfessorClaudioBonel\SENAC\UC3 - Machine Learning\2024.1\Dados'

# Carregar o arquivo
dados = pd.read_csv(endereco + r'\dados_producao_parafusos.csv')

# NECSSIDADE DE NEGOCIO: Qual seria o custo de produçao para as qtde de produçao: 12000, 15000 e 20000 parafusos?
# Funçao de primeiro grau: y = ax + b, onde y é variavel dependente e x é a variavel independente

#Isolar as variaveis como vetores
# CUSTO - Variavel estimada, logo é a variavel dependente
y_custo = dados['Custo de Produção (R$)'].values

# QTDE - Variavel que influencia/simulara a estimativa do custo, logo é a variavel independente
x_qtde = dados['Quantidade Produzida (unidades)'].values

#scikit-learn: Biblioteca para Machine Learning
# A partir dos dados de produçao de parafusos, vamos separar os dados para serem treinados e testados
from sklearn.model_selection import train_test_split

X_qtde_train, X_qtde_test, y_custo_train, y_custo_test = train_test_split(x_qtde,
                                                                          y_custo,
                                                                          test_size=0.2,
                                                                          shuffle=False)

# treinar o modelo
# importar a classe regressao linear
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()

# Treinar o modelo
# gerar a minha equaçao de primeiro grau
modelo.fit(X_qtde_train.reshape(-1,1), y_custo_train)

# Pontuaçao do modelo: Serve para verificar se o treino foi eficaz
# 0 a 1: Quanto mais proximo de 1, melhor
# VERIFICAR SE O MODELO ESTÁ BOM
score = modelo.score(X_qtde_test.reshape(-1,1), y_custo_test)

# testar o modelo nos dados de teste
# se a necesside eh prever custo(y), logo vou testar somente com os dados de qtde(x)
predicao = modelo.predict(X_qtde_test.reshape(-1,1))

# retornando a necessidade de negocio
# Qual seria o custo de produçao para as qtde de produçao: 12000, 15000 e 20000 parafusos?
qtde_produzida_simulada = np.array([12000, 15000, 20000, 25000])

# Prever o custo de produçao para as qtdes solicitadas
custo_producao_pred = modelo.predict(qtde_produzida_simulada.reshape(-1,1))

# Exibir os resultados
print(custo_producao_pred)