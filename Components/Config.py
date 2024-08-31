import streamlit as st
from PIL import Image




def Config():
    logo = Image.open('assets/logologo.jpeg')
    st.set_page_config(page_title="Auto Trainer", page_icon=logo, layout="wide", initial_sidebar_state="collapsed")
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            .css-1k0ckh2{display:none}
            .css-9s5bis{display:none}
            .css-k0sv6k{display:none}
            .css-hy8qiv {display:none}
            
            footer{visibility:hidden}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    margins_css = """
        <style>
            .main > div {
                padding-left: 0rem;
                padding-right: 0rem;
            }
        </style>
    """

    st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)
