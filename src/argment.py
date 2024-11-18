import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="pl")

    parser.add_argument("--mlflow_experiment_name", required=True, type=str)
    parser.add_argument("--mlflow_tag", required=True, type=str)

    parser.add_argument("--labeled_num", default=None, type=int)
    parser.add_argument("--batch_size", default=512, type=int)

    parser.add_argument("--model", default="CNN", type=str)
    parser.add_argument("--loss", default="PL", type=str)
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--epochs", default=100, type=int)

    parser.add_argument("--delta1_init", default=0.0, type=float)
    parser.add_argument("--delta1", default=3.0, type=float)
    parser.add_argument("--threshold1", default=10, type=int)
    parser.add_argument("--threshold2", default=70, type=int)

    return parser.parse_args()
