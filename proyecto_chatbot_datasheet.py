# Cargar librerías
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
import faiss
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk
from flask import Flask, request, render_template

# Descargar stopwords en español
nltk.download('stopwords')
nltk.download('punkt')

# Inicializar Flask
app = Flask(__name__)

# Leer el archivo de Excel
file_path = 'ex141_datasheet.xlsx'
df = pd.read_excel(file_path)

# Convertir las columnas de DataFrame a una lista de pares pregunta-respuesta
qa_pairs = df.apply(lambda row: (row[0], row[1]), axis=1).tolist()
questions, answers = zip(*qa_pairs)

# Cargar el modelo y el tokenizador
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Función para limpiar texto: eliminar símbolos y stop words
stop_words = set(stopwords.words('spanish'))

# Limpiar texto y convertir a minúsculas
def clean_text(text):
    # Eliminar símbolos
    text = re.sub(r'[^\w\s]', '', text)
    # Tokenizar y eliminar stop words
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    return ' '.join(filtered_sentence)

# Codificar las preguntas
def encode_questions(questions):
    inputs = tokenizer(list(questions), padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy()

question_embeddings = encode_questions(questions)

# Indexar las preguntas con FAISS
faiss_index = faiss.IndexFlatL2(question_embeddings.shape[1])
faiss_index.add(question_embeddings)

# Umbral de distancia para considerar una respuesta como válida
DISTANCE_THRESHOLD = 11 # Ajustar según los resultados de similitud

# Función para encontrar la respuesta más similar
def find_best_answer(query):
    query_embedding = encode_questions([query])
    D, I = faiss_index.search(query_embedding, k=1)
    if D[0][0] > DISTANCE_THRESHOLD:
        return "No entiendo la pregunta, por favor vuelve a intentar"
    return answers[I[0][0]]

# Ruta principal para la interfaz web
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["question"]
        cleaned_input = clean_text(user_input)
        response = find_best_answer(cleaned_input)
        return render_template("index.html", question=user_input, response=response)
    return render_template("index.html")

# Iniciar el script
if __name__ == "__main__":
    app.run(debug=True)
