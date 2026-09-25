# Practical 1: Text to Categorical Count Matrix (CountVectorizer)
from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "Your Service is very very bad",
    "TCS is service based company",
    "You work in bad service company"
]

counter_vectorizer = CountVectorizer()
count_matrix = counter_vectorizer.fit_transform(documents)

print("Vocabulary:", counter_vectorizer.vocabulary_)
print("\nCount Matrix (Array):\n", count_matrix.toarray())
