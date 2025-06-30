import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import streamlit as st

def main():
    st.title("Sistema de Recomendación para hacerte rico 💼")
    st.write("Este sistema te ayudará a comparar tu perfil con quienes ganan más de 50K y te dará recomendaciones.")

    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'

    if st.session_state['page'] == 'home':
        if st.button("Comenzar recomendación"):
            st.session_state['page'] = 'recomendacion'

    if st.session_state['page'] == 'recomendacion':
        # Import recomendacion page and run its main function
        from pages import recomendacion
        recomendacion.main()

if __name__ == "__main__":
    main()

# https://docs.streamlit.io/library/get-started/multipage-apps
# Local: streamlit run home_tutorial.py
# Streamlit Sharing 
# render, heroku, AWS EC2