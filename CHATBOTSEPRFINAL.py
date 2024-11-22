import os
import pandas as pd
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import re
from PIL import Image

# Configurations d'environnement
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Définir les chemins des fichiers
script_path = os.path.abspath(__file__)
base_path = os.path.dirname(script_path)

# Chemins des fichiers
csv_path = os.path.join(base_path, "CHATBOTSEPRFINAL.csv")
logo_path = os.path.join(base_path, "Logo_final.png")
response_image_path = os.path.join(base_path, "IMG_final.jpg")

# Chargement des modèles
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("fr_core_news_sm")
    except IOError as e:
        st.error(f"Erreur lors du chargement du modèle SpaCy : {e}")
        return None

@st.cache_resource
def load_sentence_transformer_model():
    try:
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle SentenceTransformer : {e}")
        return None

# Chargement des données FAQ
@st.cache_data
def load_faq_data(file_path):
    if not os.path.isfile(file_path):
        st.error(f"Erreur : le fichier '{file_path}' n'existe pas.")
        return None

    try:
        faq_data = pd.read_csv(file_path, sep=';', encoding='latin-1', on_bad_lines='warn')
        faq_data.columns = faq_data.columns.str.strip()
        if {'Questions', 'Answers'}.issubset(faq_data.columns):
            st.success("Bienvenue, comment puis-je vous aider ?")
            return faq_data
        else:
            st.error("Erreur : le fichier doit contenir les colonnes 'Questions' et 'Answers'.")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier FAQ : {e}")
        return None

# Prétraitement d'une question
def preprocess_question(question):
    doc = nlp(question.lower())
    corrected = " ".join(token.lemma_ for token in doc if not token.is_stop)
    return re.sub(r'\W+', ' ', corrected).strip()

# Calcul des embeddings des questions FAQ
@st.cache_data
def calculate_faq_embeddings(faq_data):
    return model.encode(faq_data['Questions'].fillna('').tolist(), convert_to_tensor=True)

# Trouver la question la plus similaire
def find_similar_question(question, faq_embeddings):
    question_embedding = model.encode([question], convert_to_tensor=True)
    similarities = cosine_similarity(question_embedding, faq_embeddings)
    top_index = np.argmax(similarities)
    return top_index, similarities[0][top_index]

# Raffiner la réponse
def refine_response(index, similarity_score, faq_data, threshold=0.7):
    return faq_data['Answers'].iloc[index] if similarity_score > threshold else "Désolé, je n'ai pas trouvé de réponse à votre question."

# Pipeline principal du chatbot
def chatbot_pipeline(question, faq_data, faq_embeddings):
    processed_question = preprocess_question(question)
    index, similarity_score = find_similar_question(processed_question, faq_embeddings)
    return refine_response(index, similarity_score, faq_data)

# Interface Streamlit
def main():
    st.title("Chatbot de la Société d'Environnement et de Plantation de Redeyef (SEPR)")

    # Afficher le logo en haut de la page
    if os.path.isfile(logo_path):
        logo_image = Image.open(logo_path).resize((100, 100))  # Dimensions ajustées
        st.image(logo_image, use_container_width=False)
    else:
        st.warning(f"Logo introuvable à l'emplacement : {logo_path}")

    # Charger les modèles
    global nlp, model
    nlp = load_spacy_model()
    model = load_sentence_transformer_model()

    # Charger les données FAQ
    faq_data = load_faq_data(csv_path)
    if faq_data is not None:
        faq_embeddings = calculate_faq_embeddings(faq_data)

        # Saisie de la question par l'utilisateur
        user_question = st.text_input("Posez votre question ici :")
        if st.button("Obtenir une réponse") and user_question:
            response = chatbot_pipeline(user_question, faq_data, faq_embeddings)
            st.markdown(f"**Réponse :** {response}")

            # Afficher une image associée à la réponse
            if os.path.isfile(response_image_path):
                response_image = Image.open(response_image_path).resize((400, 300))  # Dimensions ajustées
                st.image(response_image, caption="Illustration associée")
            else:
                st.warning(f"Image de réponse introuvable à l'emplacement : {response_image_path}")

if __name__ == "__main__":
    main()