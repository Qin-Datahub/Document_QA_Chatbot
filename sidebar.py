import streamlit as st
import os

def side_bar():
    with st.sidebar:
        st.markdown(
            "## Get Started\n"
            "* :key: Enter your [OpenAI API key](https://platform.openai.com/account/api-keys) below\n"
            "* :bookmark_tabs: Upload a doc (pdf or txt)\n" 
            "* :question: Ask a question regarding the document\n"
        )

        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="Paster your OpenAI API key here",
            help="You can get your API key from https://platform.openai.com/account/api-keys",
            value=os.environ.get("OPENAI_API_KEY", None) or st.session_state.get("OPENAI_API_KEY", "")
        )

        os.environ["OPENAI_API_KEY"] = api_key
        st.session_state["OPENAI_API_KEY"] = api_key
        
        if not api_key:
            st.warning(
                "Enter your OpenAI API key in the sidebar."
            )
        st.markdown("---")