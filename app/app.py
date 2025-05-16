import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import RocCurveDisplay

# Configura layout da página
st.set_page_config(page_title="Previsão de Inadimplência", layout="centered")

# Carrega o modelo
model = joblib.load("model/melhor_modelo.pkl")

# Lista das colunas que o modelo espera
colunas_modelo = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]

# Título
st.title("📊 Análise Preditiva de Inadimplência com Machine Learning")

# Introdução
st.markdown("""
Este painel analisa o risco de inadimplência de clientes com base em seus dados financeiros.  
Ao enviar uma planilha da base original (como `credit_data.csv`), o sistema aplica um modelo treinado de machine learning para prever a probabilidade de cada cliente não pagar suas dívidas no mês seguinte.

**Colunas obrigatórias:**  
`LIMIT_BAL`, `AGE`, `PAY_*`, `BILL_AMT*`, `PAY_AMT*`, etc.

---
""")

# Upload do CSV
uploaded_file = st.file_uploader("📂 Envie seu arquivo CSV da base original", type=["csv"])

if uploaded_file is not None:
    try:
        df_raw = pd.read_csv(uploaded_file)

        if not all(col in df_raw.columns for col in colunas_modelo):
            st.error("❌ O arquivo não contém todas as colunas que o modelo espera.")
            st.markdown(f"**Esperadas:** `{', '.join(colunas_modelo)}`")
        else:
            st.success("✅ Arquivo carregado com sucesso. Iniciando análise...")

            # Filtrar colunas e prever
            df = df_raw[colunas_modelo]
            previsoes = model.predict(df)
            probabilidades = model.predict_proba(df)[:, 1]

            df_resultado = df.copy()
            df_resultado['Probabilidade de Inadimplência (%)'] = (probabilidades * 100).round(2)
            df_resultado['Inadimplente (previsto)'] = ['Sim' if p == 1 else 'Não' for p in previsoes]

            inadimplentes = df_resultado['Inadimplente (previsto)'].value_counts().get('Sim', 0)
            total = len(df_resultado)
            perc = inadimplentes / total * 100

            # Texto de análise
            st.markdown(f"""
### 🧾 Resultado da Análise
Segundo a planilha enviada, o modelo previu que **{inadimplentes} clientes** não irão pagar suas dívidas no próximo mês.  
Isso representa **{perc:.2f}%** da base analisada.

Veja abaixo os **10 primeiros resultados** com a probabilidade de inadimplência estimada:
""")
            st.dataframe(df_resultado.head(10), use_container_width=True)

            # 📈 Gráfico 1 — Distribuição das probabilidades
            st.markdown("### 📊 Distribuição do Risco de Inadimplência")
            st.markdown("O gráfico abaixo mostra como as probabilidades estão distribuídas entre os clientes.")
            fig1, ax1 = plt.subplots(figsize=(6, 3))
            sns.histplot(df_resultado['Probabilidade de Inadimplência (%)'], bins=20, kde=True, color="skyblue", ax=ax1)
            ax1.set_xlabel("Probabilidade (%)")
            ax1.set_ylabel("Número de Clientes")
            ax1.set_title("Distribuição das Probabilidades de Inadimplência")
            st.pyplot(fig1)

            # 📉 Gráfico 2 — Contagem de inadimplentes
            st.markdown("### 📉 Quantidade de Inadimplentes Previstos")
            fig2, ax2 = plt.subplots(figsize=(4, 3))
            sns.countplot(x='Inadimplente (previsto)', data=df_resultado, palette=["#2ecc71", "#e74c3c"], ax=ax2)
            ax2.set_title("Classificação: Inadimplente ou Não")
            ax2.set_xlabel("Previsão do Modelo")
            ax2.set_ylabel("Total de Clientes")
            ax2.set_xticklabels(['Não', 'Sim'])
            st.pyplot(fig2)

            # 📊 Gráfico extra: inadimplência vs idade
            st.markdown("### 👵 Inadimplência por Faixa Etária")
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            sns.boxplot(x='Inadimplente (previsto)', y='AGE', data=df_resultado, palette="pastel", ax=ax3)
            ax3.set_title("Distribuição de Idade por Classificação")
            ax3.set_xlabel("Inadimplente")
            ax3.set_ylabel("Idade")
            ax3.set_xticklabels(['Não', 'Sim'])
            st.pyplot(fig3)

            # 📊 Gráfico extra: inadimplência vs limite de crédito
            st.markdown("### 💳 Inadimplência por Limite de Crédito")
            fig4, ax4 = plt.subplots(figsize=(6, 3))
            sns.boxplot(x='Inadimplente (previsto)', y='LIMIT_BAL', data=df_resultado, palette="Set2", ax=ax4)
            ax4.set_title("Distribuição do Limite de Crédito por Classificação")
            ax4.set_xlabel("Inadimplente")
            ax4.set_ylabel("Limite de Crédito (R$)")
            ax4.set_xticklabels(['Não', 'Sim'])
            st.pyplot(fig4)

            # 📥 Botão para baixar os resultados
            st.download_button(
                label="📥 Baixar CSV com as Previsões",
                data=df_resultado.to_csv(index=False).encode("utf-8"),
                file_name="resultado_previsao.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {e}")
