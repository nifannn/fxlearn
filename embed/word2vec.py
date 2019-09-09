import numpy as np
import tensorflow as tf
import logging
import time

TF_CONFIG = tf.ConfigProto(allow_soft_placement = True)

class SGNS(object):
    """
    Skip-Gram With Negative Sampling
    --------------------------------
    """
    def __init__(self, embed_dim=100, window_size=10, min_count=3, min_len=3, max_len=200,
                 random_window=False, n_sampled=100, subsample=True, subsample_thr=1e-5,
                 lr=None, name=None, log_file=None):
        self._embed_dim = embed_dim
        self._window_size = window_size
        self._min_count = min_count
        self._min_len = min_len
        self._max_len = max_len
        self._random_window = random_window
        self._n_sampled = n_sampled
        self._subsample = subsample
        self._subsample_thr = subsample_thr
        self._lr = lr
        self._set_name(name)
        self._set_logger(log_file)

    def _build_graph(self):
        tf.reset_default_graph()
        self._inputs = tf.placeholder(tf.int32, [None], name="inputs")
        self._labels = tf.placeholder(tf.int32, [None, None], name="labels")
        self._embed_matrix = tf.Variable(tf.random_uniform((len(self._ix2word), self._embed_dim), -1, 1))
        self._embed = tf.nn.embedding_lookup(self._embed_matrix, self._inputs)
        softmax_w = tf.Variable(tf.truncated_normal((len(self._ix2word), self._embed_dim), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(len(self._ix2word)))
        loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b,
                                          self._labels, self._embed,
                                          self._n_sampled, len(self._ix2word))
        self._loss = tf.reduce_mean(loss)

        if isinstance(self._lr, float) or isinstance(self._lr, int):
            self._opt = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(self._loss)
        else:
            self._opt = tf.train.AdamOptimizer().minimize(self._loss)

        self._var_init = tf.global_variables_initializer()
        self._saver = tf.train.Saver()
        self._sess = tf.Session(config=TF_CONFIG)

    def _count_words_from_file(self, filepath):
        word_count = dict()
        file_len = self._get_file_len(filepath)
        seq_generator = self._generate_seq_from_file(filepath)
        for _ in range(file_len):
            seq = next(seq_generator)
            for word in seq:
                word_count[word] = word_count.get(word, 0)+1
        return word_count

    def _generate_seq_from_file(self, filepath, delimiter=","):
        with open(filepath, "r") as f:
            line = f.readline()
            while line:
                seq = line.strip("\n").split(delimiter)
                yield seq
                line = f.readline()
        yield 0

    def _get_file_len(self, filepath):
        with open(filepath, "r") as f:
            for i, _ in enumerate(f):
                pass
        return i+1

    def _get_sample_from_seq(self, sequences):
        inputs = []
        targets = []
        for seq in sequences:
            for ix in range(len(seq)):
                window_size = np.random.randint(1, self._window_size+1) if self._random_window else self._window_size
                start = ix-window_size if (ix-window_size)>0 else 0
                end = ix + window_size
                target_words = seq[start:ix] + seq[ix+1:end+1]
                targets = targets + target_words
                inputs = inputs + [seq[ix]] * len(target_words)
        return inputs, targets

    def _preprocess_seq(self, sequences):
        result = [[self._word2ix[word] 
                   for word in seq if word in self._word2ix]
                   for seq in sequences]
        result = [seq if len(seq) <= self._max_len else seq[-self._max_len:]
                  for seq in result]
        return result

    def _save_word_vec(self, path,
                       field_separator="\t", vector_separator=" "):
        embed_dict = self.get_embeddings()
        with open(path, "w") as f:
            for word, vec in embed_dict.items():
                f.write(word+field_separator+vector_separator.join([str(v) for v in vec])+"\n")
        return self

    def _set_name(self, name):
        self.name = "Word2Vec" if name is None else name

    def _set_logger(self, log_file):
        if log_file is None:
            logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
        else:
            logging.basicConfig(filename=log_file, format='%(asctime)s %(levelname)s:%(message)s',
                            level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

    def _set_vocab(self, word_count):
        self._ix2word = [word for word, cnt in word_count.items() if cnt >= self._min_count]
        self._word2ix = {word:ix for ix, word in enumerate(self._ix2word)}

    def _train_step(self, parameter_list):
        pass

    def _train_on_seq_from_file(self, filepath, epochs, batch_size):
        word_count = self._count_words_from_file(filepath)
        self._set_vocab(word_count)
        self._build_graph()
        self._sess.run(self._var_init)
        logging.info("Start training ...")
        loss = 0
        iteration = 0
        start_time = time.time()
        his_loss = []

        for e in range(1, epochs+1):
            epoch_losses = []
            n_seq = self._get_file_len(filepath)
            seq_generator = self._generate_seq_from_file(filepath)
            for i in range(0, n_seq, batch_size):
                iteration += 1
                batch_seq = []
                for _ in range(batch_size):
                    seq = next(seq_generator)
                    if seq == 0:
                        break
                    if len(seq) >= self._min_len:
                        batch_seq.append(seq)
                batch_seq = self._preprocess_seq(batch_seq)
                batch_x, batch_y = self._get_sample_from_seq(batch_seq)
                feed = {self._inputs:batch_x,
                        self._labels:np.array(batch_y)[:, None]}
                batch_loss, _ = self._sess.run([self._loss, self._opt], feed_dict=feed)
                loss += batch_loss
                epoch_losses.append(batch_loss)

                if iteration % 1000 == 0:
                    end_time = time.time()
                    msg = "Epoch {}/{}, ".format(e, epochs)
                    msg += "Iteration: {}, ".format(iteration)
                    msg += "Avg training loss: {:.4f}, ".format(loss/1000)
                    msg += "{:.4f} sec/it".format((end_time-start_time)/1000)
                    logging.info(msg)
                    loss = 0
                    start_time = time.time()
            epoch_mean_loss = np.mean(epoch_losses)
            his_loss.append(float(epoch_mean_loss))
            msg = "Epoch {}/{}, Avg batch loss: {:.4f}".format(e, epochs, epoch_mean_loss)
            logging.info(msg)

        logging.info("Training completed, history epoch loss: {}".format(his_loss))
        return self

    def _train_on_pair(self):
        pass

    def get_embeddings(self):
        embed_matrix = self._sess.run(self._embed_matrix)
        embed_dict = {word:embed_matrix[ix].tolist()
                      for word, ix in self._word2ix.items()}
        return embed_dict

    def train(self, data, data_type="seq", source_type="file", epochs=10, batch_size=128):
        if source_type == "file":
            if data_type == "seq":
                self._train_on_seq_from_file(data, epochs, batch_size)
        return self

    def save(self, out_path, field_separator="\t", vector_separator=" "):
        self._save_word_vec(out_path, field_separator, vector_separator)



class CBOW(object):
    """CBOW
    """
    def __init__(self):
        pass
        