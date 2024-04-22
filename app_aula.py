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
    st.write('O objetivo principal deste app é .....')

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


# Abas principais
tab1, tab2 = st.tabs(["Predições", "Análise Detalhada"])

with tab1:
    if database == 'CSV':
        if file:
            Xtest = pd.read_csv(file)
            mdl_lgbm = load_model('./pickle_lgbm_pycaret')
            ypred = predict_model(mdl_lgbm, data=Xtest, raw_score=True)

            with st.expander('Visualizar CSV carregado:', expanded = False):
                c1, _ = st.columns([2,4])
                qtd_linhas = c1.slider('Visualizar quantas linhas do CSV:', 
                                        min_value = 5, 
                                        max_value = Xtest.shape[0], 
                                        step = 10,
                                        value = 5)
                st.dataframe(Xtest.head(qtd_linhas))

            with st.expander('Visualizar Predições:', expanded = True):
                c1, _, c2, c3 = st.columns([2,.5,1,1])
                treshold = c1.slider('Treshold (ponto de corte para considerar predição como True)',
                                    min_value = 0.0,
                                    max_value = 1.0,
                                    step = .1,
                                    value = .5)
                qtd_true = ypred.loc[ypred['Score_True'] > treshold].shape[0]

                c2.metric('Qtd clientes True', value = qtd_true)
                c3.metric('Qtd clientes False', value = len(ypred) - qtd_true)

            def color_pred(val):
                color = 'olive' if val > treshold else 'orangered'
                return f'background-color: {color}'

            tipo_view = st.radio('', ('Completo', 'Apenas predições'))
            if tipo_view == 'Completo':
                df_view = ypred.copy()
            else:
                df_view = pd.DataFrame(ypred.iloc[:,-1].copy())

            st.dataframe(df_view.style.applymap(color_pred, subset = ['Score_True']))

            csv = df_view.to_csv(sep = ';', decimal = ',', index = True)
            st.markdown(f'Shape do CSV a ser baixado: {df_view.shape}')
            st.download_button(label = 'Download CSV',
                                data = csv,
                                file_name = 'Predicoes.csv',
                                mime = 'text/csv')

        else:
            st.warning('Arquivo CSV não foi carregado')
    else:
        # Layout do aplicativo
        st.title('Predição de Propensão de Compra')

        # Recolher os valores das features do usuário
        accepted_cmp1 = st.number_input('AcceptedCmp1', min_value=0, max_value=1)
        accepted_cmp2 = st.number_input('AcceptedCmp2', min_value=0, max_value=1)
        accepted_cmp3 = st.number_input('AcceptedCmp3', min_value=0, max_value=1)
        accepted_cmp4 = st.number_input('AcceptedCmp4', min_value=0, max_value=1)
        accepted_cmp5 = st.number_input('AcceptedCmp5', min_value=0, max_value=1)
        age = st.number_input('Age', min_value=0)
        complain = st.number_input('Complain', min_value=0, max_value=1)
        education = st.selectbox('Education', ['Basic','Graduation','2n Cycle', 'Master', 'PhD'])
        income = st.number_input('Income', min_value=0)
        kidhome = st.number_input('Kidhome', min_value=0, max_value=10)
        marital_status = st.selectbox('Marital Status', ['Single', 'Together', 'Married', 'Divorced', 'Widow'])
        mnt_fish_products = st.number_input('MntFishProducts', min_value=0)
        mnt_fruits = st.number_input('MntFruits', min_value=0)
        mnt_gold_prods = st.number_input('MntGoldProds', min_value=0)
        mnt_meat_products = st.number_input('MntMeatProducts', min_value=0)
        mnt_sweet_products = st.number_input('MntSweetProducts', min_value=0)
        mnt_wines = st.number_input('MntWines', min_value=0)
        num_catalog_purchases = st.number_input('NumCatalogPurchases', min_value=0)
        num_deals_purchases = st.number_input('NumDealsPurchases', min_value=0)
        num_store_purchases = st.number_input('NumStorePurchases', min_value=0)
        num_web_purchases = st.number_input('NumWebPurchases', min_value=0)
        num_web_visits_month = st.number_input('NumWebVisitsMonth', min_value=0)
        recency = st.number_input('Recency', min_value=0)
        teenhome = st.number_input('Teenhome', min_value=0)
        time_customer = st.number_input('Time_Customer', min_value=0)

        # Slider para escolher o threshold
        threshold = st.slider('Escolha o Threshold', min_value=0.0, max_value=1.0, step=0.01, value=0.5)

        # Criar DataFrame com os valores inseridos pelo usuário
        user_data = pd.DataFrame({
            'AcceptedCmp1': [accepted_cmp1],
            'AcceptedCmp2': [accepted_cmp2],
            'AcceptedCmp3': [accepted_cmp3],
            'AcceptedCmp4': [accepted_cmp4],
            'AcceptedCmp5': [accepted_cmp5],
            'Age': [age],
            'Complain': [complain],
            'Education': [education],
            'Income': [income],
            'Kidhome': [kidhome],
            'Marital_Status': [marital_status],
            'MntFishProducts': [mnt_fish_products],
            'MntFruits': [mnt_fruits],
            'MntGoldProds': [mnt_gold_prods],
            'MntMeatProducts': [mnt_meat_products],
            'MntSweetProducts': [mnt_sweet_products],
            'MntWines': [mnt_wines],
            'NumCatalogPurchases': [num_catalog_purchases],
            'NumDealsPurchases': [num_deals_purchases],
            'NumStorePurchases': [num_store_purchases],
            'NumWebPurchases': [num_web_purchases],
            'NumWebVisitsMonth': [num_web_visits_month],
            'Recency': [recency],
            'Teenhome': [teenhome],
            'Time_Customer': [time_customer]
        })

        label_encoder = LabelEncoder()
        user_data['Education'] = \
            label_encoder.fit_transform(user_data['Education'])
        user_data['Marital_Status'] = \
            label_encoder.fit_transform(user_data['Marital_Status'])

        # Botão para fazer a predição
        if st.button('Prever Propensão de Compra'):
            mdl_rf = load_model('./pickle_lgbm_pycaret')
            ypred = predict_model(mdl_rf, data=user_data, raw_score=True)
            prediction_proba = mdl_rf.predict_proba(user_data)[:, 1]
            prediction = (prediction_proba > threshold).astype(int)
            st.subheader('Resultado da Predição')
            if prediction == 1:
                st.success('Este cliente é propenso a comprar o produto da campanha.')
            else:
                st.error('Este cliente não é propenso a comprar o produto da campanha.')

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
            st.subheader("Dispersão de Recency e Income")
            if 'Recency' in Xtest.columns and 'Income' in Xtest.columns and 'Age' in Xtest.columns:
                fig = px.scatter(
                    Xtest,
                    x="Recency",
                    y="Income",
                    color="Age",  # Usando Age para cor
                    color_continuous_scale=px.colors.sequential.Viridis,  # Escolhendo uma escala de cor
                    hover_name="Age",  # Mostrando a idade no hover
                    log_x=True,  # Opcional: Escala logarítmica para Recency
                    labels={"Age": "Idade"}  # Renomeando a legenda
                )
                fig.update_layout(title="Relação entre Recência e Renda Colorida por Idade")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Uma ou mais colunas necessárias não foram encontradas no DataFrame.")


        with analysis_tab3:
            st.subheader("Relação entre Renda e Idade com Cor de Classe Predita")
            if all(col in Xtest.columns for col in ['Income', 'Age']):
                chart = alt.Chart(Xtest).mark_circle().encode(
                    x=alt.X('Income:Q', title='Income'),
                    y=alt.Y('Age:Q', title='Age'),
                    color=alt.Color('Age:N', title='Age'),  # Alterei 'Predicted_Class' por 'Age' para demonstrar a cor pela idade
                    tooltip=[alt.Tooltip('Income:Q'), alt.Tooltip('Age:Q'), alt.Tooltip('Recency:Q')]
                ).properties(
                    width=800,
                    height=400,
                    title='Relationship between Income and Age'
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.error("Uma ou mais colunas necessárias não foram encontradas no DataFrame.")

          
        with analysis_tab4:
            st.subheader("Box Plot para Análise Detalhada das Características dos Clientes")
            # Removendo 'Total_Purchases' da lista
            features_to_plot = ['Income', 'MntWines']  # Adicione mais recursos aqui se necessário
        
            # Verificando se todas as colunas restantes existem
            if all(feature in Xtest.columns for feature in features_to_plot):
                for feature in features_to_plot:
                    # Verificando se a coluna é numérica
                    if pd.api.types.is_numeric_dtype(Xtest[feature]):
                        # Criando o boxplot com 'Age' no eixo x
                        fig, ax = plt.subplots(figsize=(7, 4))
                        sns.boxplot(data=Xtest, x='Age', y=feature, ax=ax, palette="deep")
                        plt.title(f'Box Plot - {feature} by Age', fontsize=14)
                        ax.set_xlabel('Age', fontsize=12)
                        ax.set_ylabel(feature, fontsize=12)
                        ax.tick_params(axis='both', which='major', labelsize=10)
                        st.pyplot(fig)
                    else:
                        st.error(f"Erro: A coluna {feature} não é numérica e não pode ser usada em um boxplot.")
            else:
                # Informando quais colunas estão faltando
                missing_columns = [col for col in features_to_plot if col not in Xtest.columns]
                st.error(f"Erro: Falta(m) a(s) seguinte(s) coluna(s) no DataFrame: {', '.join(missing_columns)}")

