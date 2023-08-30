import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Carregar a base de dados
df = pd.read_csv('registros.csv')

# Converter variáveis categóricas em numéricas usando codificação one-hot
df_encoded = pd.get_dummies(df, columns=['gênero','meta'])

# Mapear as classificações para valores numéricos 
class_mapping = {
    'intensivo': 2,
    'intermediário': 1,
    'menos_intensivo': 0
}
df_encoded['classificação'] = df_encoded['classificação'].map(class_mapping)

# Separar features e alvo
X = df_encoded.drop('classificação', axis=1)
y = df_encoded['classificação']

# Dividir em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Criar e treinar o modelo
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)

# Avaliar a precisão do modelo
y_pred = modelo.predict(X_test)
precisao = accuracy_score(y_test, y_pred)
print("Precisão do modelo:", precisao)


#------------ Teste de Previsão -----------------------
class Previsao:
    def __init__(self, genero, idade, peso, altura, meta):
        self.genero = genero
        self.idade = idade
        self.peso = peso
        self.altura = altura
        self.meta = meta
        self.classificacao = self.fazer_previsao()

    def fazer_previsao(self):
        # Carregar o modelo treinado
        modelo = DecisionTreeClassifier()
        
        # Carregar a base de dados e converter variáveis categóricas em numéricas usando codificação one-hot
        df = pd.read_csv('registros.csv')
        df_encoded = pd.get_dummies(df, columns=['gênero','meta'])
        class_mapping = {
            'intensivo': 2,
            'intermediário': 1,
            'menos_intensivo': 0
        }
        df_encoded['classificação'] = df_encoded['classificação'].map(class_mapping)
        
        # Separar features e alvo
        X = df_encoded.drop('classificação', axis=1) #Features
        y = df_encoded['classificação']  #Alvo
        
        # Treinar o modelo com os dados disponíveis
        modelo.fit(X, y)

        # Fazer a previsão usando os valores de entrada
        X_pred = pd.DataFrame({
            'idade': [self.idade],
            'peso': [self.peso],
            'altura': [self.altura],
            'gênero_feminino': [1 if self.genero == 'feminino' else 0],
            'gênero_masculino': [1 if self.genero == 'masculino' else 0],
            'meta_hipertrofia': [1 if self.meta == 'hipertrofia' else 0],
            'meta_emagrecimento': [1 if self.meta == 'emagrecimento' else 0]
        })
        previsao = modelo.predict(X_pred)
        
        return previsao[0]  # Retornar a classificação prevista
    
    def formatar_previsao(self):
        if(self.classificacao == 2):
            print("Previsão de plano escolhida é: Intensivo")
        if(self.classificacao == 1):
            print("Previsão de plano escolhida é: Intermediário")
        if(self.classificacao == 0):
            print("Previsão de plano escolhida é: Menos Intensivo")
        return self.classificacao

# Exemplo de uso
genero = "masculino"
idade = 42
peso = 55
meta = "emagrecimento"
altura = 1.75

previsao = Previsao(genero, idade, peso, meta, altura)
texto_previsao = previsao.formatar_previsao()
print("Indíce da classificação: ",texto_previsao)
