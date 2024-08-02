# Welcome to the Resume Ranker Project

-This project was built to provide scores to submitted resumes and rank them based on a similarity to a given job description.

- This project is built using BERT(Transformers Based LLM), SBERT(A more advanced version of BERT) and TF-IDF(A model that is used to find the importance of a word in a given document).

- First we will process the input text of the resume, which is retreievd by using the fitz library, and the preprocess the text using stopwords and lemmatizer.

- Next we build the BERT and SBERT model, by converting the words of the document into embeddings that is used to feed the BERT model, convert the resume input and Job description into BERT embeddings and use cosine similarity to find the similarity between the resume and the provided Job description. 

- Utilize an average of BERT score, SBERT score and TF-IDF score to generate a final score for each resume.

- Built a flask application to display these findings and rankings to the user, which also allows them to upload resumes and job descriptions.

- This project was built mainly for an internal team at my organization to streamline their hiring process and improve efficiency.
 
# Music Monsoon
