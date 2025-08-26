# Medical Question‑Answering System
This repository contains a simple retrieval‑based question‑answering (QA) system
designed to answer medical questions. The goal of the project is to
demonstrate an end‑to‑end machine learning workflow—from data
preprocessing, through model training and evaluation, to generating
example interactions with the trained system. The underlying
dataset consists of approximately 16,000 question/answer pairs
covering a broad range of medical conditions, symptoms and
treatments.
# Dataset
The primary dataset resides in mle_screening_dataset.csv and
contains two columns:
## Column	Description
## question	The medical question posed by a user
answer	A factual response describing the condition or
answering the question in plain language
Most questions occur only once, and there are slightly fewer
questions than answers due to the presence of duplicate answer
phrases. Prior to training, the data is cleaned by converting
questions to lowercase and removing punctuation. No additional
external datasets were introduced to maintain reproducibility, but
future work could easily extend the corpus with publicly available
medical QA resources.
Approach
Because each unique question has a unique answer, framing the task as
a classification problem (with one class per answer) would be
impractical. Instead, we treat the problem as information
retrieval: given a new question, we find the most similar question
in the training set and return its answer. The core components are:
TF–IDF Vectorization: We represent each question as a
TF–IDF weighted bag of 1‑gram and 2‑gram tokens. The
vectorizer is fitted on the training questions and restricted to
the 50,000 most frequent features.
Nearest Neighbor Search: During inference, the cleaned
query is transformed into a TF–IDF vector and the cosine
similarity is computed against all training question vectors.
The answer associated with the nearest question is returned.
Token‑based F1 Evaluation: For evaluation we use a simple
macro‑averaged F1 score between the gold and predicted answers.
Each answer is normalized, tokenized and compared as sets of
words. Precision is the fraction of the predicted tokens
appearing in the gold answer; recall is the fraction of gold
tokens appearing in the prediction; their harmonic mean gives
F1. While this metric penalizes synonymous phrasing, it provides
an interpretable baseline.
Results
After splitting the data into 80% training, 10% validation and 10%
test sets (without stratification due to the uniqueness of most
questions), the model achieves a mean token‑based F1 of
approximately 0.18 on the test set. This low score reflects the
heterogeneity of answers and the limitations of bag‑of‑words
matching—answers often contain additional detail beyond the core
definition that is not captured by TF–IDF. Nevertheless, in many
cases the retrieved answer is semantically appropriate.
Example interactions
Below are three illustrative questions and the answers returned by
the system:
Q: What are the symptoms of diabetes?
A: Many people with diabetes experience one or more symptoms,
including extreme thirst or hunger, a frequent need to urinate
and/or fatigue. Some lose weight without trying. Additional
signs include sores that heal slowly, dry, itchy skin, loss of
feeling or tingling in the feet and blurry eyesight. Some people
with diabetes, however, have no symptoms at all.
Q: How can I prevent glaucoma?
A: At this time, we do not know how to prevent glaucoma.
However, studies have shown that the early detection and
treatment of glaucoma, before it causes major vision loss, is the
best way to control the disease. So, if you fall into one of
the higher risk groups for the disease, make sure to have a
comprehensive dilated eye exam at least once every one to two
years.
Q: What causes hypertension?
A: Pulmonary hypertension begins with inflammation and
changes in the cells that line your pulmonary arteries. Other
factors can also affect the pulmonary arteries and cause the
condition, such as arterial wall tightening, arterial stiffness
from cell overgrowth, or blood clots. Diseases and conditions
such as mitral valve disease, chronic lung disease, sleep apnea
and sarcoidosis can contribute.
These examples illustrate how the system surfaces detailed medical
explanations when questions align well with the training data. If a
query refers to a topic absent from the corpus or uses very
different language, the system may return an unrelated answer.
Assumptions
Several simplifying assumptions underpin the project:
Vocabulary sufficiency: We assume the provided dataset
contains enough variety of terminology to answer common medical
questions. Extending the corpus with other curated medical QA
datasets (e.g. MedQuAD or health forums) could improve coverage.
Bag‑of‑words adequacy: By relying on TF–IDF and cosine
similarity, we implicitly assume that similar questions share
similar surface forms. This is not always true in practice
because synonyms and paraphrases may not share vocabulary.
Single answer mapping: Each question maps to a single
ground‑truth answer. In reality, questions can be answered in
multiple valid ways; representing multiple correct answers could
lead to better evaluation metrics.
Strengths and Weaknesses
Strengths:
Interpretability: The model is simple to train and debug.
Each prediction is traceable to a specific training question.
Efficiency: TF–IDF vectorization and nearest neighbor search
scale reasonably well to tens of thousands of examples on
commodity hardware.
Weaknesses:
Limited generalization: The system can only return answers
present in the training set. It cannot synthesize new answers or
combine multiple knowledge sources.
Sensitivity to wording: Performance degrades for
paraphrased or misspelled queries. Pre‑trained embeddings (e.g.
word2vec or transformer encoders) or character‑level models could
mitigate this sensitivity.
Evaluation mismatch: The token‑based F1 metric is a
relatively coarse measure of answer quality. Alternative
evaluation strategies (e.g. BLEU, ROUGE, or human judgment) could
better reflect semantic correctness.
Potential Improvements
Advanced Embeddings: Replace TF–IDF with pre‑trained
embeddings such as word2vec, GloVe or BERT to capture semantic
similarity beyond exact token overlap. Sentence embedding models
like Sentence‑BERT would likely improve retrieval accuracy.
Paraphrase Augmentation: Expand the training set with
paraphrased versions of existing questions using back‑translation
or paraphrase generation models. This would make the system more
robust to varied phrasing.
Hybrid Models: Combine retrieval with generative models,
e.g. using the retrieved passages as context for a
sequence‑to‑sequence model that crafts a concise answer. This
approach (called retrieval‑augmented generation, or RAG) allows
the system to provide novel answers while remaining grounded in
the corpus.
Multi‑Answer Evaluation: Group duplicate questions and treat
all associated answers as correct. Use metrics like recall@k to
better capture retrieval quality when multiple answers are valid.
User Interface: Expose the system via a web interface or
chatbot that includes clarifications and follow‑up questions to
refine the user intent.
