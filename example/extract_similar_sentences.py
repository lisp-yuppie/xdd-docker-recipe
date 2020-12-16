from os import path
import glob

import spacy
import numpy as np
import tensorflow_hub as hub


MODEL_URL = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
TARGET_SENTENCE = "long-wave radiative flux at the top of the atmosphere"
CUTOFF = 0.25


def cosine_similarity(x, y):
    len_x = np.sqrt(np.sum(x*x))
    len_y = np.sqrt(np.sum(y*y))
    return np.dot(x, y) / (len_x * len_y)

def main():
    nlp = spacy.load("en_core_web_sm")
    print("Loading model...")
    model = hub.load(MODEL_URL)
    print(f"Loaded {model}")
    print(f"Target sentence: [{TARGET_SENTENCE}]")
    [target_embedding] = model([TARGET_SENTENCE])
    print(f"Target embedding: {target_embedding.shape}")
    
    for doc in glob.glob("/input/*.txt"):
        out_file = path.join("/output/", 
                             path.basename(doc).replace(".txt", "_out.txt"))
        with open(out_file, "w") as fout:
            with open(doc) as fin:
                sentences = [sent.text for sent in nlp(fin.read()).sents]
                print(f"Got {len(sentences):,} sentences")
                sentence_embeddings = model(sentences).numpy()
                print(f"Computed {len(sentence_embeddings):,} embeddings")
                #print(f"Sample: {sentence_embeddings[:3]}")
                similarities = [cosine_similarity(target_embedding, this_embedding)
                                for this_embedding in sentence_embeddings]
                table = list(zip(sentences, similarities))
                print("Samples:")
                for i, (sent, sim) in enumerate(table[:3]):
                    print(f"{i}:  [{sent}] {sim}")
                table = [(sent,sim) for sent, sim in table if sim >= CUTOFF]
                table.sort(key=lambda x: x[1], reverse=True)
                # write results to the output directory
                fout.write("\n".join([f"{sim} {sent}" for sent, sim in table]))
                print(f"Wrote out {len(table):,} sentences to {out_file}")


if __name__ == "__main__":
    main()
