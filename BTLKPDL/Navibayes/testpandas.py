
from sklearn.feature_extraction.text import TfidfTransformer

counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],           
          [3, 2, 0],
          [3, 0, 2],
          [2, 0, 0]]
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(counts)
print(tfidf.toarray())

a = "nghia doan ngo"
b = set(a)
print(b)