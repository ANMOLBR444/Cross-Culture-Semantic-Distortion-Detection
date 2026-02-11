from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama
from langchain_community.llms.ollama import Ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in literature and culture who can convert any sentence to canonicalized form (simple english without any ambiguities and complexities), please convert the following sentence into canonicalized English."),
    ("user", "Convert the following sentence to canonicalized English: {text}"),
])

def generate_response(text):
    llm = Ollama(model="gpt-oss:120b-cloud")
    output_parser = StrOutputParser()
    chain = prompt_template|llm|output_parser
    answer = chain.invoke({"text": text})
    return answer

def generate_embedding(text):
    model = SentenceTransformer('all-mpnet-base-v2')
    embedding = model.encode(text)
    return embedding

def get_similarity(embedding1, embedding2):
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

st.title("Cross-Culture Semantic Distortion Detector")
user_input = st.text_area("Enter text to analyze:", "")
if st.button("Analyze"):
    if user_input:
        canonicalized_text = generate_response(user_input)
        st.write(f"Canonicalized Text: {canonicalized_text}")

        original_embedding = generate_embedding(user_input)
        canonicalized_embedding = generate_embedding(canonicalized_text)

        similarity_score = get_similarity(original_embedding, canonicalized_embedding)
        st.write(f"Semantic Similarity Score: {similarity_score:.4f}")
        with st.container(border=True):
            st.subheader("Result: ")
            if similarity_score >= 0.9:
                st.write("There is no semantic distortion.")
            elif similarity_score >= 0.7:
                st.write("There is some semantic distortion.")
            else:
                st.write("There is significant semantic distortion.")
    else:
        st.write("Please enter some text to analyze.")