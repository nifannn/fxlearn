import numpy as np
import pandas as pd
import sys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import util

ITEM_COL = 'item_id'
LABEL_COL = 'item_label'
N_PLOT = 1000

def plot_with_labels(low_dim_embs, labels, filename, fsize=18):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(fsize, fsize))
    for i, label in enumerate(labels):
      x, y = low_dim_embs[i, :]
      plt.scatter(x, y)
      plt.annotate(
          label,
          xy=(x, y),
          xytext=(5, 2),
          textcoords='offset points',
          ha='right',
          va='bottom')
    plt.savefig(filename)

def visualize_with_tsne(embed_file, item_label_file, out):
    item_embed = util.load_embedding(embed_file)
    item_label = pd.read_csv(item_label_file)
    item_info = item_label.set_index(ITEM_COL).to_dict('index')
    item_list = list(item_info.keys())
    item_list = np.random.choice(item_list,N_PLOT,replace=False).tolist()
    embed_matrix = np.array([item_embed[item_id] for item_id in item_list])
    norm = np.sqrt(np.sum(np.square(embed_matrix),axis=1,keepdims=True))
    normalized_embed = embed_matrix/norm
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000, method='exact')
    low_dim_embs = tsne.fit_transform(normalized_embed)
    labels = [item_info[item][LABEL_COL] for item in item_list]
    plot_with_labels(low_dim_embs, labels, out)


if __name__ == "__main__":
        run(sys.argv[1], sys.argv[2], sys.argv[3])