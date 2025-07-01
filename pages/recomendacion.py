import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
from streamlit_extras.switch_page_button import switch_page

# Ruta para guardar/leer dataset preprocesado
DATA_PATH = 'data/df_final.csv'

def scale_value(value, col_name, df):
    min_val = df[col_name].min()
    max_val = df[col_name].max()
    if max_val - min_val > 0:
        return (value - min_val) / (max_val - min_val)
    else:
        return 0

@st.cache_data
def load_and_process_data():
    if os.path.exists(DATA_PATH):
        # Si ya existe el csv preprocesado, lo cargamos directo
        df_final = pd.read_csv(DATA_PATH)
    else:
        # Si no existe, procesamos desde el raw
        df = pd.read_csv('adult.csv') 

        # Limpieza básica
        df = df.replace('?', np.nan).dropna()

        # Variable objetivo binaria
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)

        vars_importantes = ['age', 'education.num', 'hours.per.week', 'capital.gain', 'capital.loss',
                           'marital.status', 'relationship', 'workclass']

        df_filtered = df[vars_importantes + ['income']].copy()

        # One-hot encoding para categóricas
        cat_vars = ['marital.status', 'relationship', 'workclass']
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(df_filtered[cat_vars])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_vars), index=df_filtered.index)

        # Concatenar numéricas + categóricas one-hot
        df_final = pd.concat([df_filtered.drop(columns=cat_vars), encoded_df], axis=1)

        # Escalar con MinMaxScaler
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df_final.drop(columns=['income'])), 
                                 columns=df_final.drop(columns=['income']).columns, index=df_final.index)
        df_scaled['income'] = df_final['income']

        df_final = df_scaled

        # Crear carpeta si no existe
        os.makedirs('data', exist_ok=True)
        df_final.to_csv(DATA_PATH, index=False)
        
    return df_final

def user_input_features(df):
    st.sidebar.header("Tu perfil")
    age = st.sidebar.slider('Edad', 17, 90, 30)
    education_num = st.sidebar.slider('Nivel educativo (num)', 1, 16, 10)
    hours_per_week = st.sidebar.slider('Horas de trabajo por semana', 1, 99, 40)
    capital_gain_raw = st.sidebar.number_input('Ganancia de capital', min_value=0, value=0)
    capital_loss_raw = st.sidebar.number_input('Pérdida de capital', min_value=0, value=0)

    # Cargar df raw para usar en escala original de capital gain/loss
    df_raw = pd.read_csv('adult.csv').replace('?', np.nan).dropna()

    # Escalar capital gain/loss usando función definida
    capital_gain = scale_value(capital_gain_raw, 'capital.gain', df_raw)
    capital_loss = scale_value(capital_loss_raw, 'capital.loss', df_raw)

    marital_status_options = [
        'Married-civ-spouse', 'Never-married', 'Divorced', 'Separated', 'Widowed',
        'Married-spouse-absent', 'Married-AF-spouse'
    ]
    relationship_options = [
        'Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'
    ]
    workclass_options = [
        'Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov', 'Federal-gov',
        'Self-emp-inc', 'Without-pay', 'Never-worked'
    ]

    marital_status = st.sidebar.selectbox('Estado civil', marital_status_options)
    relationship = st.sidebar.selectbox('Relación familiar', relationship_options)
    workclass = st.sidebar.selectbox('Clase de trabajo', workclass_options)

    user_data = {
        'age': age,
        'education.num': education_num,
        'hours.per.week': hours_per_week,
        'capital.gain': capital_gain,
        'capital.loss': capital_loss,
    }

    user_df_num = pd.DataFrame([user_data])

    onehot_cols = [c for c in df.columns if
                   c.startswith('marital.status_') or c.startswith('relationship_') or c.startswith('workclass_')]

    user_df_cat = pd.DataFrame(0, index=[0], columns=onehot_cols)

    user_df_cat[f'marital.status_{marital_status}'] = 1
    user_df_cat[f'relationship_{relationship}'] = 1
    user_df_cat[f'workclass_{workclass}'] = 1

    user_profile = pd.concat([user_df_num, user_df_cat], axis=1)

    user_profile = user_profile[df.drop(columns=['income']).columns]

    return user_profile


def generate_recommendations(user_profile_scaled, ricos_profiles):
    recommendations = []
    for feature in ['education.num', 'hours.per.week', 'capital.gain']:
        user_val = user_profile_scaled[feature].values[0]
        avg_val = ricos_profiles[feature].mean()
        if user_val < avg_val:
            diff = round(avg_val - user_val, 2)
            recommendations.append(f"Podrías aumentar '{feature}' en aproximadamente {diff} para acercarte al perfil de quienes ganan más de 50K.")
        else:
            recommendations.append(f"Tu '{feature}' está en línea o por encima del promedio de quienes ganan más de 50K.")
    return recommendations

def main():
    st.cache_data.clear()

    st.title("Sistema de Recomendación para ganar más de 50K")

    df_final = load_and_process_data()

    user_profile = user_input_features(df_final)

    # Escalar perfil usuario basado en min/max del df_final para numéricas
    user_profile_scaled = user_profile.copy()
    cols_num = ['age', 'education.num', 'hours.per.week', 'capital.gain', 'capital.loss']
    for col in cols_num:
        min_val = df_final[col].min()
        max_val = df_final[col].max()
        if max_val - min_val > 0:
            user_profile_scaled[col] = (user_profile[col] - min_val) / (max_val - min_val)
        else:
            user_profile_scaled[col] = 0  # evitar div/0

    # Categóricas one-hot ya están 0/1

    ricos_profiles = df_final[df_final['income'] == 1].drop(columns=['income'])

    sim_scores = cosine_similarity(user_profile_scaled, ricos_profiles)[0]

    top_idx = np.argsort(sim_scores)[-3:][::-1]
    top_similares = ricos_profiles.iloc[top_idx]

    st.subheader("Tu perfil escalado:")
    st.dataframe(user_profile_scaled)

    st.subheader("Perfiles similares que ganan >50K:")
    st.dataframe(top_similares)

    st.subheader("Recomendaciones para ti:")
    recs = generate_recommendations(user_profile_scaled, ricos_profiles)
    for rec in recs:
        st.write("- " + rec)

if __name__ == "__main__":
    main()

