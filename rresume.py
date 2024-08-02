#Importing all required libraries
import fitz  # PyMuPDF
import nltk
import torch
from transformers import BertTokenizer, BertModel, SqueezeBertConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers.models.deformable_detr.image_processing_deformable_detr import max_across_indices


#Using the fitz/PyMuPDF library to extract the text from the PDF

def extract_text_from_pdf(pdf_path):
   doc = fitz.open(pdf_path)
   text = ""
   for page_num in range(doc.page_count):
       page = doc.load_page(page_num)
       text += page.get_text()
   return text

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

#Perform text processing on the attained text, which involves removing stopwords applying lemmatization(That is the concept of generating the right form/tense of the word)
#This involves converting the input text to tokens, perform the preprocessing and join the final output.
def text_processing(text):
    lemm = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())

    actual_tokens = [lemm.lemmatize(token) for token in tokens if token not in stop_words and token.isalnum()]  

    new_text = " ".join(actual_tokens)
    return new_text







def build_bert(job_description,resume):
   #Loading the BERT tokenizer and the model
   tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
   model = BertModel.from_pretrained("bert-base-uncased")

   #Let us build the BERT model which begins with coverting each word presented as an token in the BERT format and then used to feed into the model that genertes each word as an embedding.
   # All these embedings are then aggregated into a single vector of fixed size which serves as the representation of the entire text.
   # We then use the generated vector to perform cosine simalirity scores to generate a similarity between the provided resume and the job description.
   # NOTE on HIDDEN STATES : Hidden states are vectors that encode and hold information about a token and its context.
   # The BERT model adds to special token CLS and SEP to the beginning and end of each embedding represenation.
   # The BERT model either has a 12 Layer or 24 Layer Sequence.
   # The CLS token in the last layer of the model stores the hidden states of all the input tokens.

   #First let us build the method used to tokenize the input text in BERT format and generate embeddings of the text.

   def bert_embeddings(text):
      #This specifies that the output should be in the form of pytorch(pt) tensors and should have a max length of 512 and can be truncated if required.
      input_text = tokenizer(text, return_tensors='pt', truncation=True, padding = True, max_length = 512)
      #Passing the input into a BERT LLM model used to convert to embeddings
      output_embd = model(**input_text)
      #This step extracts the hidden states of each token in the text from the last layer of the BERT model and aggregates all the hidden states into a single vector. This aggregation is done averaging the hidden states over the sequence length, and further converts the vector to a numpy array and detaches it from the computation graph.
      return output_embd.last_hidden_state.mean(dim=1).detach().numpy()
   

   
   #Now we will utilize the cosine simalirity to build a function that compares the BERT embeddings of the given Job description and the provided Resume.

   def similarity_calculator(job_description,resume):
      #Building the embeddings for the Job Description
      jd_embd = bert_embeddings(text_processing(job_description))
      #Building the embeddings for the Resume
      resume_embd = bert_embeddings(text_processing(extract_text_from_pdf(resume)))
      #Comparing the two embeddings to create a cosine similarity score
      similarity_score = cosine_similarity(jd_embd,resume_embd)
      return similarity_score
   

   
   return similarity_calculator(job_description,resume)[0][0]

#This is similar to the above process, just in this case we use a hugging face transformer in the form of SBERT which is sentence bert and has a more deeper understanding of the context of the words that it is feeded with, this is utilized to get a higher accuracy in our output.
def build_sbert(job_description,resume):
   #Alternate method using SBERT(sentence bert)
    def sbert_embeddings(text):
      sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
      sbert_embeddings = sbert_model.encode(text)
      return sbert_embeddings
   
    def similarity_calculator_sbert(job_description,resume):
      #Building the embeddings for the Job Description
      jd_embd = sbert_embeddings(text_processing(job_description)).reshape(1,-1)
      #Building the embeddings for the Resume
      resume_embd = sbert_embeddings(text_processing(extract_text_from_pdf(resume))).reshape(1,-1)
      #Comparing the two embeddings to create a cosine similarity score
      similarity_score = cosine_similarity(jd_embd,resume_embd)
      return similarity_score

    return similarity_calculator_sbert(job_description,resume)[0][0]


#We also utilitze the base TF-IDF vectorizer method, this proves to be the core on which this ranker is based on, this process involves fitting the TF-IDF model with the input text, creating the TF-IDF values.
#Coverting JD and resume into the TF-IDF format and perform the cosine similarity on both these values.
def build_tfidf(job_description,resume):
   #Initialize the TfidfVectorizer
   tfidf_vect = TfidfVectorizer(max_features=5000)

   #Fit the vectorizer on the Job Description and the Resume Text, this enables the model to update vocabulary and create the TF-IDF values.
   combined_data = [text_processing(job_description)] + [text_processing(extract_text_from_pdf(resume))]
   tfidf_vect.fit(combined_data)

   #Transform the JD and the Resume text into respective TF-IDF vectors:
   jd_tfidf = tfidf_vect.transform([text_processing(job_description)])
   resume_tfidf = tfidf_vect.transform([text_processing(extract_text_from_pdf(resume))])

   #Calculate the cosine similarity
   similarity_score = cosine_similarity(jd_tfidf,resume_tfidf)
   return similarity_score[0][0]



#This function is designed to combine the three models namely => BERT , SBERT , TF-IDF and when provided with an Job description and a resume pdf file, this function will generate a score for that resume based on it's similarity with the provided job description.
def ranking(job_description, resume):
   
   score1 = build_bert(job_description, resume)
   score2 = build_sbert(job_description, resume)
   score3 = build_tfidf(job_description, resume)
   rank=0

   if score3 == 0:
      rank = 0
   else:
      rank = ((score1 + score2 + score3)/3)*100
   return rank
      
