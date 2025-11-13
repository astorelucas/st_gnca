import pandas as pd

def analisar_timestamps_csv(caminho_arquivo, nome_coluna_timestamp):
    try:
        df = pd.read_csv(caminho_arquivo)

        if nome_coluna_timestamp not in df.columns:
            print(f"Erro: A coluna '{nome_coluna_timestamp}' não foi encontrada.")
            print(f"Colunas disponíveis: {list(df.columns)}")
            return

        contagem_total = df[nome_coluna_timestamp].count()


        df[nome_coluna_timestamp] = pd.to_datetime(df[nome_coluna_timestamp], errors='coerce')

        primeiro_registro = df[nome_coluna_timestamp].min()
        ultimo_registro = df[nome_coluna_timestamp].max()

        print("--- Análise do Arquivo CSV ---")
        print(f"Arquivo: {caminho_arquivo}")
        print(f"Coluna analisada: {nome_coluna_timestamp}")
        print(f"Total de registros: {contagem_total}")
        print(f"Primeiro registro: {primeiro_registro}")
        print(f"Último registro: {ultimo_registro}")

    except FileNotFoundError:
        print(f"Erro: Arquivo não encontrado em '{caminho_arquivo}'")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


NOME_DO_ARQUIVO = 'PEMS-BAY/data-preprocessed.csv' 

NOME_DA_COLUNA = 'timestamp' 

analisar_timestamps_csv(NOME_DO_ARQUIVO, NOME_DA_COLUNA)