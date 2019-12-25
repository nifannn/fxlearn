import numpy as np
import random
import tensorflow as tf
import sys
import time
import os
import json
import tempfile

sys.path.append("../")
import util

TF_CONFIG = tf.ConfigProto(allow_soft_placement = True)

class SGNS(object):
    """
    Skip-Gram With Negative Sampling
    --------------------------------
    """

    MODELNUM = 0

    def __init__(self, embed_dim=128, window_size=10, random_window=False, distinct_targets=False,
                 item_min_count=3, item_max_count=None, item_filter_method="replace",
                 item_replace_value="<UNK>", train_with_replace_value=True,
                 seq_min_len=3, seq_max_len=200, seq_truncate_method='left',
                 n_sampled=10, ns_method="sampled_softmax", ns_remove_accidental_hits=True,
                 subsample=True, subsample_thr=1e-5,
                 lr=None, name=None, log_file=None):
        self._is_trained = False
        self._embed_dim = embed_dim
        self._window_size = window_size
        self._random_window = random_window
        self._distinct_targets = distinct_targets
        self._item_min_count = item_min_count
        self._item_max_count = item_max_count
        self._item_filter_method = item_filter_method
        self._item_replace_value = item_replace_value
        self._train_with_replace_value = train_with_replace_value
        self._seq_min_len = seq_min_len
        self._seq_max_len = seq_max_len
        self._seq_truncate_method = seq_truncate_method
        self._n_sampled = n_sampled
        self._ns_method = ns_method
        self._ns_remove_accidental_hits = ns_remove_accidental_hits
        self._subsample = subsample
        self._subsample_thr = subsample_thr
        self._lr = lr
        self._train_params = dict()
        self._set_name(name)
        self._set_logger(log_file)

    def _build_graph(self):
        tf.reset_default_graph()
        self._inputs = tf.placeholder(tf.int32, [None], name="inputs")
        self._labels = tf.placeholder(tf.int32, [None, None], name="labels")
        self._embed_matrix = tf.Variable(tf.random_uniform((len(self._ix2item), self._embed_dim), -1, 1))
        self._embed = tf.nn.embedding_lookup(self._embed_matrix, self._inputs)
        self._softmax_w = tf.Variable(tf.truncated_normal((len(self._ix2item), self._embed_dim), stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(len(self._ix2item)))
        if self._ns_method == "sampled_softmax":
            loss = tf.nn.sampled_softmax_loss(self._softmax_w, softmax_b,
                                              self._labels, self._embed,
                                              self._n_sampled, len(self._ix2item),
                                              remove_accidental_hits=self._ns_remove_accidental_hits)
        if self._ns_method == "nce":
            loss = tf.nn.nce_loss(self._softmax_w, softmax_b, 
                                  self._labels, self._embed,
                                  self._n_sampled, len(self._ix2item),
                                  remove_accidental_hits=self._ns_remove_accidental_hits)
        self._loss = tf.reduce_mean(loss)

        if isinstance(self._lr, float) or isinstance(self._lr, int):
            self._opt = tf.train.AdamOptimizer(learning_rate=self._lr).minimize(self._loss)
        else:
            self._opt = tf.train.AdamOptimizer().minimize(self._loss)

        self._var_init = tf.global_variables_initializer()
        self._saver = tf.train.Saver()
        self._sess = tf.Session(config=TF_CONFIG)

    @staticmethod
    def _check_folder(folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

    def _count_item(self, feed, feed_type, delimiter=","):
        if feed_type == "list":
            return self._count_item_from_list(feed)
        if feed_type == "file":
            return self._count_item_from_file(feed, delimiter)

    @staticmethod
    def _count_item_from_list(sequences):
        item_count = dict()
        for seq in sequences:
            for item in seq:
                item_count[item] = item_count.get(item, 0)+1
        return item_count

    @staticmethod
    def _count_item_from_file(filepath, delimiter=","):
        item_count = dict()
        with open(filepath, "r") as f:
            for line in f:
                seq = line.strip("\n").split(delimiter)
                for item in seq:
                    item_count[item] = item_count.get(item, 0)+1
        return item_count

    def _count_seq(self, feed, feed_type, delimiter=","):
        if feed_type = "list":
            return self._count_seq_from_list(feed)
        if feed_type = "file":
            return self._count_seq_from_file(feed, delimiter)

    @staticmethod
    def _count_seq_from_list(feed):
        n_seq = len(feed)
        avg_seq_len = np.mean([len(seq) for seq in feed])
        return n_seq, avg_seq_len

    @staticmethod
    def _count_seq_from_file(feed, delimiter=","):
        n_seq = 0
        total_len = 0
        with open(feed, "r") as f:
            for line in f:
                n_seq += 1
                total_len += len(line.strip("\n").split(delimiter))
        if not n_seq:
            return 0, 0
        avg_seq_len = total_len / n_seq
        return n_seq, avg_seq_len

    @staticmethod
    def _generate_list_from_file(filepath, delimiter=","):
        with open(filepath, "r") as f:
            line = f.readline()
            while line:
                res = line.strip("\n").split(delimiter)
                yield res
                line = f.readline()
        yield []

    def _generate_train_samples(self, sequences):
        inputs = []
        targets = []
        for seq in sequences:
            for ix in range(len(seq)):
                if self._item_filter_method == "replace" and not self._train_with_replace_value and seq[ix] == 0:
                    continue
                window_size = np.random.randint(1, self._window_size) if self._random_window else self._window_size
                start = ix-window_size if (ix-window_size) > 0 else 0
                end = ix + window_size
                target_idxes = seq[start:ix] + seq[ix+1:end+1]
                if self._distinct_targets:
                    target_idxes = list(set(target_idxes))
                if self._item_filter_method == "replace" and not self._train_with_replace_value:
                    target_idxes = [idx for idx in target_idxes if idx > 0]
                targets = targets + target_idxes
                inputs = inputs + [seq[ix]] * len(target_idxes)
        return inputs, targets

    def _get_ix_sequences(self, sequences):
        if self._item_filter_method == "replace":
            return [[self._item2ix[item] if item in self._item2ix else 0 for item in seq]
                     for seq in sequences]
        return [[self._item2ix[item] for item in seq if item in self._item2ix] for seq in sequences]

    @staticmethod
    def _get_datetime():
        return '_'.join(str(t) for t in time.localtime()[:5])

    @staticmethod
    def _get_file_len(filepath):
        with open(filepath, "r") as f:
            for i, _ in enumerate(f):
                pass
        return i+1

    def _get_embeddings(self, mode="embed", keep_replace_value=False):
        if mode == "embed":
            embed_matrix = self._sess.run(self._embed_matrix)
        if mode == "softmax":
            embed_matrix = self._sess.run(self._softmax_w)
        if mode == "avg":
            embed_matrix = (self._sess.run(self._embed_matrix) + self._sess.run(self._softmax_w))/2
        if self._item_filter_method == "replace" and not keep_replace_value:
            return {item:embed_matrix[ix].tolist()
                    for item, ix in self._item2ix.items() if ix > 0}
        return {item:embed_matrix[ix].tolist()
                for item, ix in self._item2ix.items()}

    def _get_model_info(self):
        model_info = {
            "name": self._name,
            "embed_dim": self._embed_dim,
            "window_size": self._window_size,
            "random_window": self._random_window,
            "distinct_targets": self._distinct_targets,
            "item_min_count": self._item_min_count,
            "item_max_count": self._item_max_count,
            "item_filter_method": self._item_filter_method,
            "item_replace_value": self._item_replace_value,
            "train_with_replace_value": self._train_with_replace_value,
            "seq_min_len": self._seq_min_len,
            "seq_max_len": self._seq_max_len,
            "seq_truncate_method": self._seq_truncate_method,
            "n_sampled": self._n_sampled,
            "ns_method": self._ns_method,
            "ns_remove_accidental_hits": self._ns_remove_accidental_hits,
            "subsample": self._subsample,
            "subsample_thr": self._subsample_thr,
            "lr": self._lr,
            "log_file": self._log_file
        }
        model_info.update(self._train_params)
        return model_info

    def _get_subsample_drop_rate(self, item_count):
        total_cnt = sum(item_count.values())
        freq = {item:cnt/total_cnt for item, cnt in item_count.items()}
        p_drop = {item:1-np.sqrt(self._subsample_thr/fr) for item, fr in freq.items()}
        return p_drop

    def _make_ix_sequences(self, feed, feed_type, delimiter=","):
        if feed_type == "list":
            return self._make_ix_sequences_from_list(feed)
        if feed_type == "file":
            res = self._make_ix_sequences_from_file(feed, delimiter)
            os.remove(feed)
            return res

    def _make_ix_seq(self, seq):
        if self._item_filter_method == "replace":
            return [self._item2ix[item] if item in self._item2ix else 0 for item in seq]
        return [self._item2ix[item] for item in seq if item in self._item2ix]

    def _make_ix_sequences_from_list(self, sequences):
        return [self._make_ix_seq(seq) for seq in sequences]

    def _make_ix_sequences_from_file(self, filepath, delimiter=","):
        tmp_file = tempfile.mkstemp(prefix="tmp_w2v_",text=True)[1]
        with open(tmp_file, "w") as o:
            with open(filepath, "r") as f:
                for line in f:
                    seq = line.strip("\n").split(delimiter)
                    ix_seq = self._make_ix_seq(seq)
                    if ix_seq:
                        o.write(delimiter.join([str(i) for i in ix_seq])+"\n")
        return tmp_file

    @classmethod
    def _new_name(cls):
        name = "Item2Vec_{}".format(cls.MODELNUM)
        cls.MODELNUM += 1
        return name

    def _subsampling(self, sequences, item_count):
        p_drop = self._get_subsample_drop_rate(item_count)
        if self._item_filter_method == "replace":
            return [[ix if ix > 0 and random.random() < 1-p_drop[self._ix2item[ix]]
                     else 0
                     for ix in seq]
                     for seq in sequences]
        return [[ix for ix in seq if random.random() < 1-p_drop[self._ix2item[ix]]]
                 for seq in sequences]

    def _subsampling_with_p_drop(self, sequences, p_drop):
        if self._item_filter_method == "replace":
            return [[ix if ix > 0 and random.random() < 1-p_drop[self._ix2item[ix]]
                     else 0
                     for ix in seq]
                     for seq in sequences]
        return [[ix for ix in seq if random.random() < 1-p_drop[self._ix2item[ix]]]
                 for seq in sequences]

    def _truncate_sequences(self, feed, feed_type, delimiter=","):
        if feed_type == "list":
            return self._truncate_sequences_from_list(feed)
        if feed_type == "file":
            return self._truncate_sequences_from_file(feed, delimiter)

    def _truncate_sequences_from_file(self, feed, delimiter=","):
        tmp_file = tempfile.mkstemp(prefix="tmp_w2v_",text=True)[1]
        with open(tmp_file, "w") as o:
            with open(feed, "r") as f:
                for line in f:
                    seq = line.strip("\n").split(delimiter)
                    truncated_seq = self._truncate_seq(seq)
                    if truncated_seq:
                        o.write(",".join(truncated_seq)+"\n")
        return tmp_file

    def _truncate_sequences_from_list(self, sequences):
        res = []
        for seq in sequences:
            truncated_seq = self._truncate_seq(seq)
            if truncated_seq:
                res.append(truncated_seq)
        return res

    def _truncate_seq(self, seq):
        if len(seq) < self._seq_min_len:
            return []
        if isinstance(self._seq_max_len, int) or isinstance(self._seq_max_len, float):
            if len(seq) > self._seq_max_len:
                if self._seq_truncate_method == 'left':
                    return seq[-self._seq_max_len:]
                if self._seq_truncate_method == 'right':
                    return seq[:self._seq_max_len]
        return seq

    def _train(self, feed, feed_type, epochs=10, batch_size=128,
              shuffle=True, verbose=False, item_pools=None, delimiter=","):
        self._logger.info("preprocessing data ...")
        if verbose:
            self._logger.info("truncating sequences ...")
        res = self._truncate_sequences(feed, feed_type, delimiter)
        if verbose:
            n_seq, avg_seq_len = self._count_seq(res, feed_type, delimiter)
            self._logger.info("{} sequences, average length: {}".format(n_seq, avg_seq_len))
            self._logger.info("filtering items ...")
        item_count = self._count_item(res, feed_type, delimiter)
        item_count = self._set_item_pools(item_count, True, item_pools=item_pools)
        if verbose:
            self._logger.info("{} items for training ".format(len(item_count)))
        res = self._make_ix_sequences(res, feed_type, delimiter)
        if self._subsample:
            res = self._subsampling(res, item_count, feed_type, delimiter)
        self._logger.info("generating training samples ...")

    def _train_list(self, feed, epochs=10, batch_size=128,
                    shuffle=True, verbose=False, item_pools=None):
        self._logger.info("preprocessing data ...")
        if verbose:
            self._logger.info("truncating sequences ...")
        sequences = self._truncate_sequences(feed)
        if verbose:
            self._logger.info("{} sequences, average length: {}".format(len(sequences),
                                                                        np.mean([len(seq) for seq in sequences])))
            self._logger.info("filtering items ...")
        item_count = self._count_item(sequences)
        item_count = self._set_item_pools(item_count, True, item_pools=item_pools)
        if verbose:
            self._logger.info("{} items for training".format(len(item_count)))
        sequences = self._get_ix_sequences(sequences)
        if self._subsample:
            sequences = self._subsampling(sequences, item_count)
        self._logger.info("building tensorflow graph ...")
        self._build_graph()
        X, y = self._generate_train_samples(sequences)
        if verbose:
            self._logger.info("{} training samples".format(len(X)))
        self._train_model(X, y, epochs, batch_size, shuffle, verbose)
        return self

    def _train_file(self, feed, epochs=10, batch_size=128,
                    shuffle=True, verbose=False, item_pools=None):
        self._logger.info("preprocessing data ...")
        if verbose:
            self._logger.info()
        item_count = self._count_item_from_file(feed)
        if verbose:
            self._logger.info("{} items in file after truncating sequences".format(len(item_count)))
        item_count = self._set_item_pools(item_count, True, item_pools=item_pools)
        if self._subsample:
            p_drop = self._get_subsample_drop_rate(item_count)
        if verbose:
            self._logger.info("{} items for training".format(len(item_count)))
        self._build_graph()
        self._sess.run(self._var_init)
        self._logger.info("Start training ...")
        loss = 0
        iteration = 0
        start_time = time.time()
        his_loss = []

        for e in range(1, epochs+1):
            epoch_losses = []
            n_seq = self._get_file_len(feed)
            generator = self._generate_list_from_file(feed)
            for i in range(0, n_seq, batch_size):
                iteration += 1
                batch_seq = []
                for _ in range(batch_size):
                    seq = next(generator)
                    if not seq:
                        break
                    batch_seq.append(seq)
                batch_seq = self._truncate_sequences(batch_seq)
                batch_seq = self._get_ix_sequences(batch_seq)
                if self._subsample:
                    batch_seq = self._subsampling_with_p_drop(batch_seq, p_drop)
                batch_x, batch_y = self._generate_train_samples(batch_seq)
                feed_data = {self._inputs:batch_x,
                             self._labels:np.array(batch_y)[:,None]}
                batch_loss, _ = self._sess.run([self._loss, self._opt], feed_dict=feed_data)
                loss += batch_loss
                epoch_losses.append(batch_loss)

                if iteration % 1000 == 0 and verbose:
                    end_time = time.time()
                    msg = "Epoch {}/{}, ".format(e, epochs)
                    msg += "Iteration: {}, ".format(iteration)
                    msg += "Avg training loss: {:.4f}, ".format(loss/1000)
                    msg += "{:.4f} sec/it".format((end_time-start_time)/1000)
                    self._logger.info(msg)
                    loss = 0
                    start_time = time.time()
            epoch_mean_loss = np.mean(epoch_losses)
            his_loss.append(float(epoch_mean_loss))
            msg = "Epoch {}/{}, Avg batch loss: {:.4f}".format(e, epochs, epoch_mean_loss)
            self._logger.info(msg)
        self._logger.info("Training completed, history epoch loss: {}".format(his_loss))
        self._is_trained = True
        self._train_params = {"epochs": epochs, "batch_size": batch_size,
                              "shuffle": shuffle, "history_loss": his_loss}
        return self

    def _train_model(self, X, y, epochs, batch_size, shuffle, verbose):
        self._sess.run(self._var_init)
        self._logger.info("Start training ...")
        loss = 0
        iteration = 0
        start_time = time.time()
        his_loss = []

        for e in range(1, epochs+1):
            if shuffle:
                pairs = list(zip(X, y))
                np.random.shuffle(pairs)
                X, y = zip(*pairs)
            epoch_losses = []
            n_sample = len(X)
            for start in range(0, n_sample, batch_size):
                iteration += 1
                batch_x = X[start:start+batch_size]
                batch_y = y[start:start+batch_size]

                feed = {self._inputs:batch_x,
                        self._labels:np.array(batch_y)[:,None]}
                train_loss, _ = self._sess.run([self._loss, self._opt], feed_dict=feed)
                loss += train_loss
                epoch_loss.append(train_loss)

                if iteration % 1000 == 0 and verbose:
                    end_time = time.time()
                    msg = "Epoch {}/{}, ".format(e, epochs)
                    msg += "Iteration: {}, ".format(iteration)
                    msg += "Avg training loss: {:.4f}, ".format(loss/1000)
                    msg += "{:.4f} sec/it".format((end_time-start_time)/1000)
                    self._logger.info(msg)
                    loss = 0
                    start_time = time.time()
            epoch_mean_loss = np.mean(epoch_losses)
            his_loss.append(float(epoch_mean_loss))
            msg = "Epoch {}/{}, Avg batch loss: {:.4f}".format(e, epochs, epoch_mean_loss)
            self._logger.info(msg)
        self._logger.info("Training completed, history epoch loss: {}".format(his_loss))
        self._is_trained = True
        self._train_params = {"epochs": epochs, "batch_size": batch_size,
                              "shuffle": shuffle, "history_loss": his_loss}
        return self

    def _save_model_info(self, folder):
        model_info = self._get_model_info()
        path = os.path.join(folder, self._get_datetime()+"_model_info.txt")
        with open(path, "w") as out:
            json.dump(model_info, out)

    def _save_item_vec(self, folder, mode="embed", save_replace_value=False,
                       field_separator="\t", vector_separator=" "):
        embed_dict = self._get_embeddings(mode, save_replace_value)
        path = os.path.join(folder, self._get_datetime()+"_embeddings_{}.txt".format(mode))
        with open(path, "w") as f:
            for item, vec in embed_dict.items():
                f.write(str(item)+field_separator+vector_separator.join([str(v) for v in vec])+"\n")

    def _set_name(self, name):
        self._name = self._new_name() if name is None else name

    def _set_logger(self, log_file):
        self._log_file = log_file
        self._logger = util.logger.Logger(log_file, self._name)

    def _set_item_pools(self, item_count, return_new_count=False, item_pools=None):
        self._ix2item = list(item_count.keys())
        if isinstance(self._item_max_count, int) or isinstance(self._item_max_count, float):
            self._ix2item = [item for item in self._ix2item if item_count[item] <= self._item_max_count]
        if isinstance(self._item_min_count, int) or isinstance(self._item_min_count, float):
            self._ix2item = [item for item in self._ix2item if item_count[item] >= self._item_min_count]
        if item_pools:
            self._ix2item = [item for item in self._ix2item if item in item_pools]
        new_count = {item:item_count[item] for item in self._ix2item if item in item_count}
        if self._item_filter_method == "replace":
            self._ix2item = [self._item_replace_value] + self._ix2item
        self._item2ix = {item:ix for ix, item in enumerate(self._ix2item)}
        if return_new_count:
            return new_count

    @property
    def name(self):
        return self._name

    @property
    def embed_dim(self):
        return self._embed_dim

    @property
    def n_items(self):
        return len(self._ix2item)

    @property
    def model_info(self):
        return self._get_model_info()
    
    def save(self, folder, mode="embed", save_replace_value=False, field_separator="\t", vector_separator=" "):
        if not self._is_trained:
            self._logger.warn("Model has not been trained!")
        self._check_folder(folder)
        self._save_model_info(folder)
        if mode in ["embed", "softmax", "avg"]:
            self._save_item_vec(folder, mode=mode, save_replace_value=save_replace_value, field_separator=field_separator, vector_separator=vector_separator)
        else:
            for m in ["embed", "softmax", "avg"]:
                self._save_item_vec(folder, mode=m, save_replace_value=save_replace_value, field_separator=field_separator, vector_separator=vector_separator)
        self._logger.info("Saved in {} successfully.".format(folder))

    def train(self, feed, feed_type="list", epochs=10, batch_size=128,
              shuffle=True, verbose=False, item_pools=None, delimiter=","):
        self._train(feed=feed, feed_type=feed_type, epochs=epochs, batch_size=batch_size,
                    shuffle=shuffle, verbose=verbose, item_pools=item_pools, delimiter=delimiter)
        return self

        
    

        
        