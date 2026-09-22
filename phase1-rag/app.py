import streamlit as st

from query import answer_question

st.title("Smart Supply Chain Assistant")

question = st.text_input("Ask a question:")

if st.button("Ask Question"):
    if question.strip():
        response = answer_question(question)

        st.write("### Answer")
        st.write(response["answer"])

        st.write("### Sources")

        for document in response["context"]:
            st.write(document.page_content)
    else:
        st.write("Please enter a question.")