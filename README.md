## Proyecto de Chatbot utilizando modelos transformadores y FAISS (Facebook AI Similarity Search)

Los modelos transformadores son una clase de redes neuronales que son especialmente efectivas para el procesamiento del lenguaje natural (PNL); estos modelos pueden aprender patrones complejos en el lenguaje y usarlos para generar respuestas precisas.

FAISS es una biblioteca de búsqueda de similitud de vectores de Facebook AI; permite la indexación y la búsqueda rápida de grandes conjuntos de vectores, lo que es esencial para la recuperación de respuestas eficiente.

##
El proyecto consiste en procesar la información de un datasheet (Router TP-Link EX141) mediante técnicas de NLP (limpieza, tokenización, vectorización, etc.) y el resultado se almacena en estructuras multidimensionales (preguntas y respuestas); posteriormente cada pregunta que ingresa el usuario en la interfaz Web del Chatbot, se almacena en un vector dentro de la estructura multidimensional y se calcula la menor distancia Euclidiana con el resto de vectores para encontrar la pregunta que tenga mayor similutud y así generar la respuesta adecuada.
