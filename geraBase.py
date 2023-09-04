import csv
import random

def generate_csv(filename, num_records):
    # Abrir o arquivo CSV em modo de escrita
    with open(filename, 'w', newline='') as csvfile:
        # Definir os nomes das colunas
        fieldnames = ['gênero', 'idade', 'peso', 'altura','meta', 'restricao','classificação']

        # Inicializar o escritor CSV
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Escrever o cabeçalho no arquivo CSV
        writer.writeheader()

        # Gêneros com 90% masculino e 10% feminino
        gender_choices = ['masculino'] * 3 + ['feminino']
        meta_choices = ['hipertrofia'] + ['emagrecimento']
        # Gerar registros no loop
        for _ in range(num_records):
            genero = random.choice(gender_choices)
            idade = random.randint(18, 40)
            peso = random.randint(65, 110)
            altura = round(random.uniform(1.5, 2.0), 2)
            meta = random.choice(meta_choices)
            classificacao = 'intensivo'

            # Escrever o registro no arquivo CSV
            writer.writerow({
                'gênero': genero,
                'idade': idade,
                'peso': peso,
                'altura': altura,
                'meta': meta,
                'restricao': "nenhuma",
                'classificação': classificacao
            })

    print(f'{num_records} registros foram gerados e salvos no arquivo {filename}')

# Número de registros a serem gerados
num_records = 1000000

# Nome do arquivo CSV
filename = 'base_dados.csv'

# Chamar o método para gerar o arquivo CSV
generate_csv(filename, num_records)


# # adição de linhas do Plano Intensivo
# def add_rows_to_csv(filename, num_records_to_add):
#     # Gêneros com 90% masculino e 10% feminino
#     gender_choices = ['masculino'] * 6 + ['feminino']
#     meta_choices = ['hipertrofia'] + ['emagrecimento']
    
#     # Abrir o arquivo CSV em modo de adição
#     with open(filename, 'a', newline='') as csvfile:
#         # Definir os nomes das colunas
#         fieldnames = ['gênero', 'idade', 'peso', 'altura','meta', 'classificação']

#         # Inicializar o escritor CSV
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         # Adicionar registros no loop
#         for _ in range(num_records_to_add):
#             genero = random.choice(gender_choices)
#             idade = random.randint(18, 40)
#             peso = random.randint(70, 100)
#             altura = round(random.uniform(1.5, 2.0), 2)
#             meta = random.choice(meta_choices)
#             classificacao = 'intensivo'

#             # Escrever o registro no arquivo CSV
#             writer.writerow({
#                 'gênero': genero,
#                 'idade': idade,
#                 'peso': peso,
#                 'altura': altura,
#                 'meta': meta,
#                 'classificação': classificacao
#             })

#     print(f'{num_records_to_add} registros foram adicionados ao arquivo {filename}')

# # Número de registros a serem adicionados
# num_records_to_add = 1000000

# # Nome do arquivo CSV existente
# filename = 'registros2.csv'

# # Chamar o método para adicionar linhas ao arquivo CSV
# add_rows_to_csv(filename, num_records_to_add)

# adição de linhas do Plano Intermeidiário
# def add_rows_to_csv(filename, num_records_to_add):
#     gender_choices = ['masculino'] * 2 + ['feminino']
#     meta_choices = ['hipertrofia'] + ['emagrecimento']
#     restricao = ['nenhum']
#     # Abrir o arquivo CSV em modo de adição
#     with open(filename, 'a', newline='') as csvfile:
#         # Definir os nomes das colunas
#         fieldnames = ['gênero', 'idade', 'peso', 'altura','meta', 'restricao','classificação']
#         # Inicializar o escritor CSV
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         # Adicionar registros no loop
#         for _ in range(num_records_to_add):
#             genero = random.choice(gender_choices)
#             idade = random.randint(40, 64)
#             peso = random.randint(50, 100)
#             meta = random.choice(meta_choices)
#             altura = round(random.uniform(1.4, 1.9), 2)
#             classificacao = 'intermediário'

#             # Escrever o registro no arquivo CSV
#             writer.writerow({
#                 'gênero': genero,
#                 'idade': idade,
#                 'peso': peso,
#                 'altura': altura,
#                 'meta': meta,
#                 'restricao': "nenhuma",
#                 'classificação': classificacao
#             })

#     print(f'{num_records_to_add} registros foram adicionados ao arquivo {filename}')

# # Número de registros a serem adicionados
# num_records_to_add = 1000000

# # Nome do arquivo CSV existente
# filename = 'base_dados.csv'

# # Chamar o método para adicionar linhas ao arquivo CSV
# add_rows_to_csv(filename, num_records_to_add)

# # # adição de linhas do Plano Menos Intensivo
# def add_rows_to_csv(filename, num_records_to_add):
#     gender_choices = ['feminino'] * 2 + ['masculino']
#     meta_choices = ['hipertrofia'] + ['emagrecimento']
#     restricao = ['outro'] + ['diabetes'] + ['doenca_respiratoria'] + ['problema_fisico']
#     # Abrir o arquivo CSV em modo de adição
#     with open(filename, 'a', newline='') as csvfile:
#         # Definir os nomes das colunas
#         fieldnames = ['gênero', 'idade', 'peso', 'altura', 'meta','restricao','classificação']

#         # Inicializar o escritor CSV
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#         # Adicionar registros no loop
#         for _ in range(num_records_to_add):
#             genero = random.choice(gender_choices)
#             idade = random.randint(15, 75)
#             peso = random.randint(40, 83)
#             altura = round(random.uniform(1.4, 1.8), 2)
#             meta = random.choice(meta_choices)
#             classificacao = 'menos_intensivo'

#             # Escrever o registro no arquivo CSV
#             writer.writerow({
#                 'gênero': genero,
#                 'idade': idade,
#                 'peso': peso,
#                 'altura': altura,
#                 'meta': meta,
#                 'restricao': restricao,
#                 'classificação': classificacao
#             })

#     print(f'{num_records_to_add} registros foram adicionados ao arquivo {filename}')

# # Número de registros a serem adicionados
# num_records_to_add = 1000000

# # Nome do arquivo CSV existente
# filename = 'base_dados.csv'

# # Chamar o método para adicionar linhas ao arquivo CSV
# add_rows_to_csv(filename, num_records_to_add)


