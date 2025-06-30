import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Carga dataset preprocesado (ya con one-hot y escalado)
@st.cache_data
def load_preprocessed_data(path='data/df_final.csv'):
    df = pd.read_csv(path)
    return df

# Variables importantes 
vars_importantes = [
    'age', 'education.num', 'hours.per.week', 'capital.gain', 'capital.loss',
    'marital.status_Married-civ-spouse', 'marital.status_Never-married', 
    'relationship_Husband', 'relationship_Not-in-family',
    'workclass_Private', 'workclass_Self-emp-inc'
]

def get_similar_profiles(user_vector, df_encoded, top_k=3):
    """
    Devuelve los perfiles top_k más similares a user_vector (coseno)
    solo entre los que ganan >50K
    """
    ricos = df_encoded[df_encoded['income'] == 1]
    ricos_X = ricos[vars_importantes]

    sim_scores = cosine_similarity(user_vector, ricos_X)[0]
    top_indices = np.argsort(sim_scores)[-top_k:][::-1]
    return ricos_X.iloc[top_indices], sim_scores[top_indices]

def generate_recommendations(user_vector, similar_profiles, feature_names):
    """
    Genera recomendaciones comparando user_vector con perfiles similares
    """
    recommendations = []

    # función para obtener valor de feature en vector usuario
    def get_user_val(feature):
        idx = feature_names.get_loc(feature)
        return user_vector[0][idx]

    # Comparar algunos features clave
    for feature in ['education.num', 'hours.per.week', 'capital.gain']:
        user_val = get_user_val(feature)
        avg_val = similar_profiles[feature].mean()
        if avg_val > user_val:
            recommendations.append(
                f"Podrías mejorar '{feature}' de {user_val:.1f} a aproximadamente {avg_val:.1f}."
            )
        else:
            recommendations.append(f"Tu '{feature}' está en línea con perfiles exitosos.")

    # Ejemplo con workclass (puedes ajustar o extender)
    workclass_feats = ['workclass_Private', 'workclass_Self-emp-inc']
    for f in workclass_feats:
        user_val = get_user_val(f)
        avg_val = similar_profiles[f].mean()
        if avg_val > user_val:
            rec = f"Tener empleo en '{f.replace('workclass_', '')}' está relacionado con mayores ingresos."
            recommendations.append(rec)

    return recommendations