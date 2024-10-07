from pydantic import BaseModel
# text = f"""You are an AI assistant specialized in document retrieval tasks. Given an image of a document page, your task is to generate retrieval queries that someone might use to find this document in a large corpus.
#     Your task is to create retrieval queries in {language} for this document content at different levels of complexity and ambiguity.\n"""

#     text2 = """Please generate 3 retrieval queries based on the content of the document:

#     1. A simple, straightforward query
#     2. A complex query requiring understanding of multiple aspects of the document
#     3. An ambiguous query that could retrieve this document among others

#     For each query, provide a brief explanation of its complexity level or ambiguity and why it would be effective or challenging for retrieval.

#     Format your response as a JSON object with the following structure:

#     {
#     "simple_query": "Your query here",
#     "simple_query_answer": "Answer to the simple query",
#     "simple_explanation": "Brief explanation",
#     "complex_query": "Your query here",
#     "complex_query_answer": "Answer to the complex query",
#     "complex_explanation": "Brief explanation",
#     "ambiguous_query": "Your query here",
#     "ambiguous_query_answer": "Answer to the ambiguous query",
#     "ambiguous_explanation": "Brief explanation"
#     }

#     Generate the queries based on this image and provide the response in the specified JSON format."""


class Prompt:
    def __init__(self, prompt: str, model: BaseModel):
        self.prompt = prompt
        self.schema = model


FRENCH_PROMPT = """Vous êtes un assistant IA spécialisé dans les tâches de recherche de documents. Étant donné une image d'une page de document, votre tâche est de générer des requêtes de recherche qu'une personne pourrait utiliser pour trouver ce document dans un large corpus.
Votre tâche est de créer des requêtes de recherche en FRANCAIS pour le contenu de ce document à différents niveaux de complexité et d'ambiguïté.
Veuillez générer 3 requêtes de recherche basées sur le contenu du document :

1. Une requête simple et directe
2. Une requête complexe nécessitant la compréhension de plusieurs aspects du document
3. Une requête ambiguë qui pourrait retrouver ce document parmi d'autres

Pour chaque requête, fournissez une brève explication de son niveau de complexité ou d'ambiguïté et pourquoi elle serait efficace ou difficile pour la recherche.

Formatez votre réponse comme un objet JSON avec la structure suivante :
{
"simple_query": "Votre requête ici",
"simple_query_answer": "Réponse à la requête simple",
"simple_explanation": "Brève explication",
"complex_query": "Votre requête ici",
"complex_query_answer": "Réponse à la requête complexe",
"complex_explanation": "Brève explication",
"ambiguous_query": "Votre requête ici",
"ambiguous_query_answer": "Réponse à la requête ambiguë",
"ambiguous_explanation": "Brève explication"
}

Générez les requêtes basées sur cette image et fournissez la réponse dans le format JSON spécifié."""


class QuestionAnswer(BaseModel):
    simple_query: str
    simple_query_answer: str
    simple_explanation: str
    complex_query: str
    complex_query_answer: str
    complex_explanation: str
    ambiguous_query: str
    ambiguous_query_answer: str
    ambiguous_explanation: str
