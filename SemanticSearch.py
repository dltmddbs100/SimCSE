import numpy as np

from torch import nn


def semantic_search(top_k, args, tokenizer, model, corpus, queries):

  model.to(args['device'])
  model.eval()

  cos=nn.CosineSimilarity()

  encodings=tokenizer.batch_encode_plus(corpus,padding='max_length',max_length=50,truncation=True,return_tensors='pt')
  corpus_embeddings=model.encode(encodings,args['device'])

  for query in queries:
    query_embedding = model.encode(tokenizer.encode_plus(query,padding='max_length',max_length=50,truncation=True,return_tensors='pt'), args['device'])
    
    cos_scores = cos(corpus_embeddings, query_embedding).cpu().detach().numpy()

    top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

    print("\n======================\n")
    print("Query:", query)
    print(f"\nTop {top_k} most similar sentences in corpus:\n")

    for idx in top_results[0:top_k]:
        print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))