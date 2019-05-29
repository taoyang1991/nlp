# encoding=utf-8

"""
该文件提供一些与数据无关的工具类函数
"""
import os
import sys
import logging
import tensorflow as tf

reload(sys)
sys.setdefaultencoding("utf-8")


def create_path(file_path):
    if not tf.gfile.IsDirectory(file_path):
        tf.gfile.MakeDirs(file_path)


def get_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def export_model(sess, model, path, version, char_to_id):
    export_path = os.path.join(path, str(version))
    if tf.gfile.IsDirectory(export_path):
        tf.gfile.DeleteRecursively(export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(model.input_data)
    tensor_info_dropout = tf.saved_model.utils.build_tensor_info(model.keep_prob)
    tensor_info_length = tf.saved_model.utils.build_tensor_info(model.input_length)
    tensor_info_batch_size = tf.saved_model.utils.build_tensor_info(model.batch_size)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(model.predict_label)

    predict_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'sentences': tensor_info_x, "dropout": tensor_info_dropout, "sentences_length": tensor_info_length,
                    "batch_size": tensor_info_batch_size},
            outputs={'label': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'predict_label':
                predict_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                predict_signature,
        },
        legacy_init_op=legacy_init_op
    )
    builder.save()

    with tf.gfile.GFile(os.path.join(export_path, "review_classify_char_to_id.csv"), "w") as file:
        for key, value in char_to_id.iteritems():
            file.write("%s\t%s\n" % (key, value))
