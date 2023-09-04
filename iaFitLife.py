import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Carregar a base de dados
df = pd.read_csv('base_dados.csv')

# Converter variáveis categóricas em numéricas usando codificação one-hot, ex: 1: true e 0: False
df_encoded = pd.get_dummies(df, columns=['gênero','meta','restricao'])

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

# Salvar o modelo treinado em um arquivo
joblib.dump(modelo, 'modelo_decision_tree.pkl')


#------------ Teste de Previsão -----------------------
class Previsao:
    def __init__(self, genero, idade, peso, altura, meta, restricao):
        self.genero = genero
        self.idade = idade
        self.peso = peso
        self.altura = altura
        self.meta = meta
        self.restricao = restricao
        self.classificacao = self.fazer_previsao()

    def fazer_previsao(self):
        # # Carregar o modelo treinado
        # modelo = DecisionTreeClassifier()
        # Carregar o modelo treinado a partir do arquivo
        modelo = joblib.load('modelo_decision_tree.pkl')
        
        # Carregar a base de dados
        df = pd.read_csv('base_dados.csv')
        
        # converte variáveis categóricas em numéricas usando codificação one-hot, em 1 True e 0 False
        df_encoded = pd.get_dummies(df, columns=['gênero','meta','restricao'])
        
        # Mapeia os níveis de classificação
        class_mapping = {
            'intensivo': 2,
            'intermediário': 1,
            'menos_intensivo': 0
        }
        # Convertermos as classificações conforme o class_mapping
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
            'meta_emagrecimento': [1 if self.meta == 'emagrecimento' else 0],
            'meta_hipertrofia': [1 if self.meta == 'hipertrofia' else 0],
            'restricao_diabetes': [1 if self.restricao == 'diabetes' else 0],
            'restricao_doenca_respiratoria': [1 if self.restricao == 'doenca_respiratoria' else 0],
            'restricao_nenhuma': [1 if self.restricao == 'nenhuma' else 0],
            'restricao_outro': [1 if self.restricao == 'outro' else 0],
            'restricao_problema_fisico': [1 if self.restricao == 'problema_físico' else 0]
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


# Dados para teste de Previsão
genero = "masculino"
idade = 40
peso = 55
meta = "hipertrofia"
altura = 1.75
restricao = "problema_fisico"

# Executando uma previsão com os dados de teste
previsao = Previsao(genero, idade, peso, altura, meta,restricao)
texto_previsao = previsao.formatar_previsao()
print("Indíce da classificação: ",texto_previsao)
