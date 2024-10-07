from pydantic import BaseModel

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

BASE_PROMT = """You are an assistant specialized in Multimodal RAG tasks.

The task is the following: given an image from a pdf page, you will have to generate questions that can be asked by a user to retrieve information from a large documentary corpus.

The question should be relevant to the page, and should not be too specific or too general. The question should be about the subject of the page, and the answer needs to be found in the page.

Remember that the question is asked by a user to get some information from a large documentary corpus that contains multimodal data. Generate a question that could be asked by a user without knowing the existence and the content of the corpus.

Generate as well the answer to the question, which should be found in the page. And the format of the answer should be a list of words answering the question.

Generate at most ONE pair of query and answer per page in a dictionary with the following format, answer ONLY this dictionary NOTHING ELSE:


{
    "question": "XXXXXX",
    "answer": "YYYYYY"
}


where XXXXXX is the question and 'YYYYYY' is the corresponding  answer that could be as long as needed.

Note: If there are no questions to ask about the page, return an empty list. Focus on making relevant questions concerning the page.

Here is the page:"""

class BaseQuery(BaseModel):
    question: str
    answer: str

base_prompt = Prompt(BASE_PROMT, BaseQuery)