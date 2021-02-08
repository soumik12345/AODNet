import subprocess
import streamlit as st
from aodnet.inferer import Inferer


def main():
    st.markdown(
        '<h1 align="center">AOD-Net: All-in-One Dehazing Network</h1><hr>',
        unsafe_allow_html=True
    )
    subprocess.run(['rm', '-r', './checkpoints/'])
    inferer = Inferer()
    st.sidebar.text('Building Model...')
    try:
        inferer.build_model()
    except FileNotFoundError:
        subprocess.run(['apt-get', 'unzip'])
        inferer.build_model()
    st.sidebar.text('Done!')
    uploaded_files = st.sidebar.file_uploader(
        'Upload Images', accept_multiple_files=True
    )
    if len(uploaded_files) > 0:
        for uploaded_file in uploaded_files:
            original_image, prediction = inferer.infer(image_path=uploaded_file)
            col_1, col_2 = st.beta_columns(2)
            with col_1:
                st.image(original_image, caption='Hazy Image')
            with col_2:
                st.image(prediction, caption='Predicted Image')
            st.markdown('___')
    subprocess.run(['rm', '-r', './checkpoints'])


if __name__ == '__main__':
    main()
