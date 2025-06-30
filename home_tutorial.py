import streamlit as st
from streamlit_extras.switch_page_button import switch_page

def main():
    st.title("Sistema de Recomendación para hacerte rico 💼")
    st.write("Este sistema te ayudará a comparar tu perfil con quienes ganan más de 50K y te dará recomendaciones.")

    if st.button("Comenzar recomendación"):
        st.session_state['page'] = 'recomendacion'

    # If page state is recomendacion, import and run that page's main()
    if st.session_state.get('page') == 'recomendacion':
        import recomendacion
        recomendacion.main()

if __name__ == "__main__":
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'
    main()

# https://docs.streamlit.io/library/get-started/multipage-apps
# Local: streamlit run home_tutorial.py
# Streamlit Sharing 
# render, heroku, AWS EC2