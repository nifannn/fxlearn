from . import config

def download_embedding():
    pass

def load_embedding():
    goods_embedding = dict()
    with open(config.embedding_path, "r") as f:
        for line in f.readlines():
            goods_id = line.split("\t")[0]
            vec = line.split("\t")[-1].strip("\n").split()
            vec = [float(v) for v in vec]
            goods_embedding[goods_id] = vec
    return goods_embedding