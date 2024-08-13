import numpy as np
from sentence_transformers import SentenceTransformer

class VectorDB():
    def __init__(self, doc, model):
        self.doc = doc
        self.vector = {}
        self.model = model
        vectors = self.model.encode(self.doc)
        for i, doc in enumerate(self.doc):
            self.vector[doc] = vectors[i]

    def cosine_similarity(self, u, v):
        product = np.dot(u, v)
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        return product / (norm_u * norm_v)
    
    def get_top(self, query, n=5):
        similarity_score={}
        query_vector=self.model.encode([query])[0]
        for key in self.vector:
            similarity_score[key] = self.cosine_similarity(query_vector, self.vector[key])
        print(similarity_score)
        return sorted(similarity_score.items(), key=lambda x: x[1], reverse=True)[:n]
    
doc = ["I like apples", "I like pears", "I like dogs", "I like cats"]
model = SentenceTransformer("all-MiniLM-L6-v2")
db = VectorDB(doc, model)
print(db.get_top("fruit", n=2))
