from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Carregue o modelo treinado
modelo = joblib.load('modelo_decision_tree.pkl')

@app.route('/previsao', methods=['POST'])
def fazer_previsao():
    try:
        data = request.get_json()
        genero = data['genero']
        idade = data['idade']
        peso = data['peso']
        altura = data['altura']
        meta = data['meta']
        restricao = data['restricao']

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

        # Fazer a previsão usando o modelo
        X_pred = pd.DataFrame({
            'idade': [idade],
            'peso': [peso],
            'altura': [altura],
            'gênero_feminino': [1 if genero == 'feminino' else 0],
            'gênero_masculino': [1 if genero == 'masculino' else 0],
            'meta_emagrecimento': [1 if meta == 'emagrecimento' else 0],
            'meta_hipertrofia': [1 if meta == 'hipertrofia' else 0],
            'restricao_diabetes': [1 if restricao == 'diabetes' else 0],
            'restricao_doenca_respiratoria': [1 if restricao == 'doenca_respiratoria' else 0],
            'restricao_nenhuma': [1 if restricao == 'nenhuma' else 0],
            'restricao_outro': [1 if restricao == 'outro' else 0],
            'restricao_problema_fisico': [1 if restricao == 'problema_fisico' else 0]
        })

        previsao = modelo.predict(X_pred)

        # Mapear a classificação de volta para o texto
        class_mapping_inverso = {
            2: 'intensivo',
            1: 'intermediário',
            0: 'menos_intensivo'
        }
        texto_previsao = class_mapping_inverso[previsao[0]]

        return jsonify({'previsao': texto_previsao})
    except Exception as e:
        return jsonify({'erro': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
