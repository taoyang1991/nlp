# encoding=utf-8

"""
该文件提供一些load数据, 处理数据集的函数
"""
import pickle
import tensorflow as tf
import sys
import numpy as np
import re

reload(sys)
sys.setdefaultencoding("utf-8")

cats_lable = {
    '0': [1, 0],
    '1': [0, 1]
}

cats_name = ["非品牌", "品牌"]
cats_dic = {
    '0': "非品牌",
    "1": "品牌",
}
split_word = "\t"


def get_vocab_to_id(file_path, vocab_save_path, reload=True):
    if not reload and tf.gfile.Exists(vocab_save_path):
        print("load vocab to id from file")
        reader = tf.gfile.GFile(vocab_save_path, "r")
        vocab_to_id = pickle.load(reader)
    else:
        vocab_to_id_set = set(list(["<UNK>", "<PAD>"]))
        sentences = SentenceGenerator(file_path)
        for line in sentences():
            line = line.strip().decode("utf-8")
            line_split = line.split(split_word)
            if len(line_split) != 2:
                continue
            cat,review = line_split
            vocab_to_id_set |= set(list(review.strip()))
        vocab_to_id = {char: idx for idx, char in enumerate(list(vocab_to_id_set))}

        with tf.gfile.GFile(vocab_save_path, "w") as f:
            f.write(pickle.dumps(vocab_to_id))
    return vocab_to_id


class Batch():
    def __init__(self):
        self.inputs = []
        self.input_lengths = []
        self.labels = []


class SentenceGenerator(object):
    def __init__(self, path):
        self.path = path
        self.length = None

    def __len__(self):
        if self.length:
            return self.length
        num = 0
        for _ in tf.gfile.GFile(self.path, "r"):
            num += 1
        self.length = num
        return self.length

    def __call__(self):
        num = 0
        for line in tf.gfile.GFile(self.path, "r"):
            num += 1
            yield line
        self.length = num


class DataHelper(object):
    def __init__(self, vocab_to_id):
        self.vocab_to_id = vocab_to_id
        self.pad_char_id = vocab_to_id.get("<PAD>")
        self.unk_char_id = vocab_to_id.get("<UNK>")

    def sentence_to_vec(self, sentence_list, is_padding=False, max_len=500, padding_direction='after'):
        text_vec = []
        for idx, char in enumerate(sentence_list):
            text_vec.append(self.vocab_to_id.get(char, self.unk_char_id))

        if is_padding:
            if len(text_vec) > max_len:
                text_vec = text_vec[:max_len]
            else:
                pad_list = [self.pad_char_id] * (max_len - len(text_vec))
                if padding_direction == 'after':
                    text_vec.extend(pad_list)
                else:
                    pad_list.extend(text_vec)
                    text_vec = pad_list
        return text_vec

    def get_sentence_label(self, cat_id):
        if cat_id in cats_lable:
            return cats_lable.get(cat_id)
        else:
            return cats_lable.get("ot")

    def get_cats_name(self, idx):
        return cats_name[idx]

    def get_cats_name_by_ids(self, sidx):
        return cats_dic[sidx]

    def create_prediction_batch(self, sen_list):
        batch = Batch()
        for sen in sen_list:
            sen = sen.decode("utf-8")
            sen_list = list(sen)
            sen_vec = self.sentence_to_vec(sen_list)
            batch.inputs.append(sen_vec)
            batch.input_lengths.append(len(sen_vec))
        batch.inputs = self.numpy_fillpad(batch.inputs)
        return batch

    def numpy_fillpad(self, data):
        # Get lengths of each row of data
        lens = np.array([len(i) for i in data])

        # Mask of valid places in each row
        mask = np.arange(lens.max()) < lens[:, None]

        # Setup output array and put elements from data into masked positions
        out = np.full(mask.shape, self.pad_char_id)
        out[mask] = np.concatenate(data)
        return out

    def clean_emoji(self, restr, desstr=""):
        try:
            co = re.compile(u'[\U00010000-\U0010ffff]')
        except re.error:
            co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
        return co.sub(desstr, restr)

    def clean_other(self, restr, desstr=" "):
        return restr.replace("\\\\n", desstr).replace("\\\\r", desstr)

    def split_review(self, review):
        review = review.decode("utf-8")
        pat = re.compile(u"[。，！？,!?~ ;；()（）～]+")
        split_list = re.split(pat, review)
        return filter(lambda x: len(x) > 0, split_list)


class BatchManager(object):
    def __init__(self, data_generator, batch_size, vocab_to_id):
        self.data_generator = data_generator()
        self.batch_size = batch_size
        self.vocab_to_id = vocab_to_id
        self.data_helper = DataHelper(vocab_to_id)

    def getBatches(self):
        batch = Batch()
        for idx, sen in enumerate(self.data_generator):
            sen = sen.strip().decode("utf-8")
            sen_split = sen.split(split_word)
            if len(sen_split) != 2:
                continue
            cat_id,sentence = sen_split
            sentence_list = list(sentence)
            sen_vec = self.data_helper.sentence_to_vec(sentence_list)
            batch.inputs.append(sen_vec)
            batch.input_lengths.append(len(sen_vec))
            batch.labels.append(self.data_helper.get_sentence_label(cat_id))
            if (idx + 1) % self.batch_size == 0:
                batch.inputs = self.data_helper.numpy_fillpad(batch.inputs)
                yield batch
                batch = Batch()

    def get_all_data_to_batch(self):
        batch = Batch()
        for idx, sen in enumerate(self.data_generator):
            sen = sen.strip().decode("utf-8")
            sen_split = sen.split(split_word)
            if len(sen_split) != 2:
                continue
            cat_id,sentence = sen_split
            sentence_list = list(sentence)
            sen_vec = self.data_helper.sentence_to_vec(sentence_list)
            batch.inputs.append(sen_vec)
            batch.input_lengths.append(len(sen_vec))
            batch.labels.append(self.data_helper.get_sentence_label(cat_id))
        batch.inputs = self.data_helper.numpy_fillpad(batch.inputs)
        return batch


if __name__ == "__main__":
    vocab_to_id = get_vocab_to_id("data/train_data", "data/vocab_to_id")
    data_generator = SentenceGenerator("data/train_data")
    batchManage = BatchManager(data_generator, 100, vocab_to_id)
    a = 0
    for i in batchManage.getBatches():
        print(i.inputs)
        print(i.labels)
        print(i.input_lengths)
        a += 1
        if a > 10:
            exit(0)
