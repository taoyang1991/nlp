# encoding=utf-8

import sys
import argparse
from model_usage import ModelUsage

reload(sys)
sys.setdefaultencoding("utf-8")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        # 是否训练
        "--train",
        type=bool,
        default=False
    )
    parser.add_argument(
        # 是否导出模型
        "--export",
        type=bool,
        default=False
    )
    parser.add_argument(
        ## 字向量
        "--embedding_size",
        type=int,
        default=300
    )
    parser.add_argument(
        "--rnn_size",
        type=int,
        default=256
    )
    parser.add_argument(
        ## 几种分类
        "--class_num",
        type=int,
        default=2
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--epoch_num",
        type=int,
        default=20
    )
    parser.add_argument(
        "--epoch_step",
        type=int,
        default=10
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model/",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/model.ckpt",
    )
    parser.add_argument(
        "--summary_path",
        type=str,
        default="summary/",
    )
    parser.add_argument(
        "--vocab_file",
        type=str,
        default="data/vocab_to_id",
    )
    parser.add_argument(
        "--cats_file",
        type=str,
        default="data/cats",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/train_data",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="data/test_data",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="log/",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="",
    )
    FLAGS, unparsed = parser.parse_known_args()
    usage = ModelUsage(FLAGS)
    usage.run()
