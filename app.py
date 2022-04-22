import streamlit as st
import streamlit.components.v1 as components
from lime_explainer import explainer, tokenizer, METHODS


def format_dropdown_labels(val):
    return METHODS[val]['name']

# Define page settings
st.beta_set_page_config(
    page_title='LIME Explainer App',
    # layout="wide"
)

# Build app
title_text = 'Interactive LIME Explainer'

st.markdown(f"<h2 style='text-align: center;'><b>{title_text}</b></h2>", unsafe_allow_html=True)

st.text("")
input_text = st.text_input('Enter your text:', "")
n_samples = st.text_input('Number of samples to generate for LIME explainer: (For really long input text, go up to 5000)', value=1000)
method_list = tuple(label for label, val in METHODS.items())
method = st.selectbox(
    'Choose Dataset:',
    method_list,
    index=4,
    format_func=format_dropdown_labels,
)

if st.button("Explain Results"):
    with st.spinner('Calculating...'):
        text = tokenizer(input_text)
        exp = explainer(method,
                        path_to_file=METHODS[method]['file'],
                        text=text,
                        lowercase=METHODS[method]['lowercase'],
                        num_samples=int(n_samples))
        # Display explainer HTML object
        components.html(exp.as_html(), height=800)
