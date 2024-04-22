import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.classification import *
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import altair as alt

# Configuração inicial da página
st.set_page_config(page_title='Simulador - Case Ifood',
                   page_icon='./images/logo_fiap.png',
                   layout='wide',
                   initial_sidebar_state='expanded')

st.title('Simulador - Conversão de Vendas')

# Descrição do App
with st.expander('Descrição do App', expanded=False):
    st.write('O objetivo principal deste app é analisar a propensão de compra dos clientes com base em diversas características.')

# Sidebar com informações e escolha do tipo de entrada
with st.sidebar:
    c1, c2 = st.columns(2)
    c1.image('./images/logo_fiap.png', width=100)
    c2.write('')
    c2.subheader('Auto ML - Fiap [v1]')

    database = st.radio('Fonte dos dados de entrada (X):', ('CSV', 'Online'))
    if database == 'CSV':
        st.info('Upload do CSV')
        file = st.file_uploader('Selecione o arquivo CSV', type='csv')
        if file:
            Xtest = pd.read_csv(file)
        else:
            Xtest = None

# Abas principais
tab1, tab2 = st.tabs(["Predições", "Análise Detalhada"])

with tab1:
    if database == 'CSV':
        if file and Xtest is not None:
            # Carregamento / instanciamento do modelo pkl
            mdl_lgbm = load_model('./pickle_lgbm_pycaret')
            # Predict do modelo
            ypred = predict_model(mdl_lgbm, data=Xtest, raw_score=True)

            with st.expander('Visualizar CSV carregado:', expanded=False):
                qtd_linhas = st.slider('Visualizar quantas linhas do CSV:', 
                                       min_value=5, 
                                       max_value=Xtest.shape[0], 
                                       step=10, 
                                       value=5)
                st.dataframe(Xtest.head(qtd_linhas))

            with st.expander('Visualizar Predições:', expanded=True):
                treshold = st.slider('Treshold (ponto de corte para considerar predição como True)',
                                     min_value=0.0,
                                     max_value=1.0,
                                     step=.1,
                                     value=.5)
                qtd_true = ypred.loc[ypred['Score_True'] > treshold].shape[0]

                st.metric('Qtd clientes True', value=qtd_true)
                st.metric('Qtd clientes False', value=len(ypred) - qtd_true)

                def color_pred(val):
                    color = 'olive' if val > treshold else 'orangered'
                    return f'background-color: {color}'

                tipo_view = st.radio('', ('Completo', 'Apenas predições'))
                if tipo_view == 'Completo':
                    df_view = ypred.copy()
                else:
                    df_view = pd.DataFrame(ypred.iloc[:,-1].copy())

                st.dataframe(df_view.style.applymap(color_pred, subset=['Score_True']))

                csv = df_view.to_csv(sep=';', decimal=',', index=True)
                st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
                st.download_button(label='Download CSV',
                                   data=csv,
                                   file_name='Predicoes.csv',
                                   mime='text/csv')
        else:
            st.warning('Arquivo CSV não foi carregado')
    else:
        # Implementação para o modo 'Online' vem aqui.

with tab2:
    if database == 'CSV' and file and Xtest is not None:
        st.header("Análise Detalhada")
        analysis_tab1, analysis_tab2, analysis_tab3, analysis_tab4 = st.tabs(
            ["Gráfico de Dispersão Plotly 1", "Gráfico de Dispersão Plotly 2", "Gráfico Altair", "Box Plot Seaborn"])

        with analysis_tab1:
            st.subheader("Relação entre Recency e Income")
            fig = px.scatter(
                Xtest,
                x="Recency",
                y="Income",
                color="Income",
                color_continuous_scale="reds"
            )
            st.plotly_chart(fig, use_container_width=True)

        with analysis_tab2:
            st.subheader("Dispersão de Recency, Income e Total Purchases")
            fig = px.scatter(
                Xtest,
                x="Recency",
                y="Income",
                size="Total_Purchases",
                color="Predicted_Class",
                color_continuous_scale=px.colors.sequential.Viridis,
                hover_name="Age",
                log_x=True,
                size_max=60,
                labels={"Predicted_Class": "Propensity to Buy"}
            )
            st.plotly_chart(fig, use_container_width=True)

        with analysis_tab3:
            st.subheader("Relação entre Renda e Idade com Cor de Classe Predita")
            chart = alt.Chart(Xtest).mark_circle().encode(
                x='Income',
                y='Age',
                color='Predicted_Class:N',
                tooltip=['Income', 'Age', 'Recency', 'Total_Purchases', 'Predicted_Class']
            ).properties(
                width=800,
                height=400,
                title='Relationship between Income and Age with Predicted Class Color'
            )
            st.altair_chart(chart, use_container_width=True)

        with analysis_tab4:
            st.subheader("Box Plot para Análise Detalhada das Características dos Clientes")
            features_to_plot = ['Income', 'Age', 'Total_Purchases', 'MntWines'] 
            for feature in features_to_plot:
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.boxplot(data=Xtest, x='Predicted_Class', y=feature, ax=ax, palette="deep")
                plt.title(f'Box Plot - {feature} by Predicted Class', fontsize=14)
                ax.set_xlabel('Predicted Class', fontsize=12)
                ax.set_ylabel(feature, fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=10)
                st.pyplot(fig)
