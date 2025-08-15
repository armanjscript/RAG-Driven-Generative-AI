from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import textwrap
import spacy
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from collections import Counter
import numpy as np
from langchain_ollama import OllamaLLM
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage

llm = OllamaLLM(model="qwen2.5:latest")


query = "define a rag store"

db_records = [
    "Retrieval Augmented Generation (RAG) represents a sophisticated hybrid approach in the field of artificial intelligence, particularly within the realm of natural language processing (NLP).",
    "It innovatively combines the capabilities of neural network-based language models with retrieval systems to enhance the generation of text, making it more accurate, informative, and contextually relevant.",
    "This methodology leverages the strengths of both generative and retrieval architectures to tackle complex tasks that require not only linguistic fluency but also factual correctness and depth of knowledge.",
    "At the core of Retrieval Augmented Generation (RAG) is a generative model, typically a transformer-based neural network, similar to those used in models like GPT (Generative Pre-trained Transformer) or BERT (Bidirectional Encoder Representations from Transformers).",
    "This component is responsible for producing coherent and contextually appropriate language outputs based on a mixture of input prompts and additional information fetched by the retrieval component.",
    "Complementing the language model is the retrieval system, which is usually built on a database of documents or a corpus of texts.",
    "This system uses techniques from information retrieval to find and fetch documents that are relevant to the input query or prompt.",
    "The mechanism of relevance determination can range from simple keyword matching to more complex semantic search algorithms which interpret the meaning behind the query to find the best matches.",
    "This component merges the outputs from the language model and the retrieval system.",
    "It effectively synthesizes the raw data fetched by the retrieval system into the generative process of the language model.",
    "The integrator ensures that the information from the retrieval system is seamlessly incorporated into the final text output, enhancing the model's ability to generate responses that are not only fluent and grammatically correct but also rich in factual details and context-specific nuances.",
    "When a query or prompt is received, the system first processes it to understand the requirement or the context.",
    "Based on the processed query, the retrieval system searches through its database to find relevant documents or information snippets.",
    "This retrieval is guided by the similarity of content in the documents to the query, which can be determined through various techniques like vector embeddings or semantic similarity measures.",
    "The retrieved documents are then fed into the language model.",
    "In some implementations, this integration happens at the token level, where the model can access and incorporate specific pieces of information from the retrieved texts dynamically as it generates each part of the response.",
    "The language model, now augmented with direct access to retrieved information, generates a response.",
    "This response is not only influenced by the training of the model but also by the specific facts and details contained in the retrieved documents, making it more tailored and accurate.",
    "By directly incorporating information from external sources, Retrieval Augmented Generation (RAG) models can produce responses that are more factual and relevant to the given query.",
    "This is particularly useful in domains like medical advice, technical support, and other areas where precision and up-to-date knowledge are crucial.",
    "Retrieval Augmented Generation (RAG) systems can dynamically adapt to new information since they retrieve data in real-time from their databases.",
    "This allows them to remain current with the latest knowledge and trends without needing frequent retraining.",
    "With access to a wide range of documents, Retrieval Augmented Generation (RAG) systems can provide detailed and nuanced answers that a standalone language model might not be capable of generating based solely on its pre-trained knowledge.",
    "While Retrieval Augmented Generation (RAG) offers substantial benefits, it also comes with its challenges.",
    "These include the complexity of integrating retrieval and generation systems, the computational overhead associated with real-time data retrieval, and the need for maintaining a large, up-to-date, and high-quality database of retrievable texts.",
    "Furthermore, ensuring the relevance and accuracy of the retrieved information remains a significant challenge, as does managing the potential for introducing biases or errors from the external sources.",
    "In summary, Retrieval Augmented Generation represents a significant advancement in the field of artificial intelligence, merging the best of retrieval-based and generative technologies to create systems that not only understand and generate natural language but also deeply comprehend and utilize the vast amounts of information available in textual form.",
    "A RAG vector store is a database or dataset that contains vectorized data points."
]

def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        use_idf=True,
        norm='l2',
        ngram_range=(1, 2),  # Use unigrams and bigrams
        sublinear_tf=True,   # Apply sublinear TF scaling
        analyzer='word'      # You could also experiment with 'char' or 'char_wb' for character-level features
    )
    tfidf = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return similarity[0][0]

def print_formatted_response(response):
    # Define the width for wrapping the text
    wrapper = textwrap.TextWrapper(width=80)  # Set to 80 columns wide
    wrapped_text = wrapper.fill(text=response)

    # Print the formatted response with a header and footer
    print("\nResponse:")
    print("---------------")
    print(wrapped_text)
    print("---------------\n")


nlp = spacy.load("en_core_web_sm")

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def preprocess_text(text):
    doc = nlp(text.lower())
    lemmatized_words = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        lemmatized_words.append(token.lemma_)
    return lemmatized_words

def expand_with_synonyms(words):
    expanded_words = words.copy()
    for word in words:
        expanded_words.extend(get_synonyms(word))
    return expanded_words

def calculate_enhanced_similarity(text1, text2):
    # Preprocess and tokenize texts
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)

    # Expand with synonyms
    words1_expanded = expand_with_synonyms(words1)
    words2_expanded = expand_with_synonyms(words2)

    # Count word frequencies
    freq1 = Counter(words1_expanded)
    freq2 = Counter(words2_expanded)

    # Create a set of all unique words
    unique_words = set(freq1.keys()).union(set(freq2.keys()))

    # Create frequency vectors
    vector1 = [freq1[word] for word in unique_words]
    vector2 = [freq2[word] for word in unique_words]

    # Convert lists to numpy arrays
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)

    # Calculate cosine similarity
    cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    return cosine_similarity


def call_llm_with_full_text(user_query):
    try:
        # Create the message chain
        messages = [
            SystemMessage(content="You are an expert Natural Language Processing exercise expert."),
            AIMessage(content="1. You can explain read the input and answer in detail"),
            HumanMessage(content=user_query)
        ]
        
        # Invoke the LLM
        response = llm.invoke(messages)
        return response.strip()
    except Exception as e:
        return str(e)

##Vector Search
def find_best_match(text_input, records):
    best_score = 0
    best_record = None
    for record in records:
        current_score = calculate_cosine_similarity(text_input, record)
        if current_score > best_score:
            best_score = current_score
            best_record = record
    return best_score, best_record

best_similarity_score, best_matching_record = find_best_match(query, db_records)
print_formatted_response(best_matching_record)

# Metrics
print(f"Best Cosine Similarity Score: {best_similarity_score:.3f}")

# Enhanced Similarity
response = best_matching_record
print(query,": ", response)
similarity_score = calculate_enhanced_similarity(query, best_matching_record)
print(f"Enhanced Similarity:, {similarity_score:.3f}")

# Augmented Input
augmented_input=query+": "+best_matching_record
print_formatted_response(augmented_input)

# Generation
llm_response = call_llm_with_full_text(augmented_input)
print_formatted_response(llm_response)


##Index-based Search

# Search Function
def setup_vectorizer(records):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(records)
    return vectorizer, tfidf_matrix

def find_best_match(query, vectorizer, tfidf_matrix):
    query_tfidf = vectorizer.transform([query])
    similarities = cosine_similarity(query_tfidf, tfidf_matrix)
    best_index = similarities.argmax()  # Get the index of the highest similarity score
    best_score = similarities[0, best_index]
    return best_score, best_index

vectorizer, tfidf_matrix = setup_vectorizer(db_records)

best_similarity_score, best_index = find_best_match(query, vectorizer, tfidf_matrix)
best_matching_record = db_records[best_index]

print_formatted_response(best_matching_record)

# Cosine Similarity
print(f"Best Cosine Similarity Score: {best_similarity_score:.3f}")
print_formatted_response(best_matching_record)

# Enhanced Similarity
response = best_matching_record
print(query,": ", response)
similarity_score = calculate_enhanced_similarity(query, response)
print(f"Enhanced Similarity:, {similarity_score:.3f}")

# Feature Extraction
def setup_vectorizer(records):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(records)

    # Convert the TF-IDF matrix to a DataFrame for display purposes
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

    # Display the DataFrame
    print(tfidf_df)

    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = setup_vectorizer(db_records)

print(vectorizer, tfidf_matrix)

# Augmented Input
augmented_input=query+": "+best_matching_record
print_formatted_response(augmented_input)

# Generation
llm_response = call_llm_with_full_text(augmented_input)
print_formatted_response(llm_response)


##Modular RAG

class RetrievalComponent:
    def __init__(self, method='vector'):
        self.method = method
        if self.method == 'vector' or self.method == 'indexed':
            self.vectorizer = TfidfVectorizer()
            self.tfidf_matrix = None

    def fit(self, records):
      self.documents = records  # Initialize self.documents here
      if self.method == 'vector' or self.method == 'indexed':
        self.tfidf_matrix = self.vectorizer.fit_transform(records)

    def retrieve(self, query):
        if self.method == 'keyword':
            return self.keyword_search(query)
        elif self.method == 'vector':
            return self.vector_search(query)
        elif self.method == 'indexed':
            return self.indexed_search(query)

    def keyword_search(self, query):
        best_score = 0
        best_record = None
        query_keywords = set(query.lower().split())
        for index, doc in enumerate(self.documents):
            doc_keywords = set(doc.lower().split())
            common_keywords = query_keywords.intersection(doc_keywords)
            score = len(common_keywords)
            if score > best_score:
                best_score = score
                best_record = self.documents[index]
        return best_record

    def vector_search(self, query):
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)
        best_index = similarities.argmax()
        return db_records[best_index]

    def indexed_search(self, query):
        # Assuming the tfidf_matrix is precomputed and stored
        query_tfidf = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_tfidf, self.tfidf_matrix)
        best_index = similarities.argmax()
        return db_records[best_index]
    
# Modular RAG Strategies
retrieval = RetrievalComponent(method='vector')  # Choose from 'keyword', 'vector', 'indexed'
retrieval.fit(db_records)
best_matching_record = retrieval.retrieve(query)

print_formatted_response(best_matching_record)


# Cosine Similarity
print(f"Best Cosine Similarity Score: {best_similarity_score:.3f}")
print_formatted_response(best_matching_record)

# Enhanced Similarity
response = best_matching_record
print(query,": ", response)
similarity_score = calculate_enhanced_similarity(query, response)
print("Enhanced Similarity:", similarity_score)

# Augmented Input
augmented_input=query+ " "+ best_matching_record
print_formatted_response(augmented_input)

# Generation
llm_response = call_llm_with_full_text(augmented_input)
print_formatted_response(llm_response)
