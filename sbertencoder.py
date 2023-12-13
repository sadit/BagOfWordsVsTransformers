from sentence_transformers import SentenceTransformer
import h5py
import json
import gzip
import sys
import os

class TextEncoder:
    def __init__(self, modelname='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', modelnick="sbert-multi-L12-v2"):
        self.modelname = modelname
        self.modelnick = modelnick
        self.model = SentenceTransformer(modelname)

    def encode(self, text):
        return self.model.encode(text)

def json_tweets(filename, output=None, key="text"):
    T = []

    print(f"** reading {filename} dataset")
    if filename.endswith(".gz"):
        with gzip.open(filename, 'rt') as f:
            for line in f:
                tweet = json.loads(line)
                text = tweet[key]
                # do some preprocessing?
                T.append(text)
    else:
        with open(filename) as f:
            for line in f:
                tweet = json.loads(line)
                text = tweet[key]
                # do some preprocessing?
                T.append(text)

    assert len(T) > 0, f"ERROR {filename} is empty"
    print(T[:3])

    E = TextEncoder()
    print(f"** encoding with {E.modelname}")
    emb = E.encode(T)
    if output is None:
        output = os.path.basename(filename).replace(".json.gz", "").replace(".json", "")
        output = output + f"--{E.modelnick}.h5"
    
    print(f"** saving embeddings in {output}")
    with h5py.File(output, "w") as f:
        f.attrs['filename'] = filename
        f.attrs['modelname'] = E.modelname 
        f.attrs['modelnick'] = E.modelnick
        f.create_dataset("emb", emb.shape, dtype=emb.dtype)[:] = emb
 
