import config
import models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='BiLSTM', help='name of the model')
parser.add_argument('--save_name', type=str)

parser.add_argument('--train_prefix', type=str, default='dev_train')
parser.add_argument('--test_prefix', type=str, default='dev_dev')
parser.add_argument('--seed', type=int, default=66)

args = parser.parse_args()
model = {
    'LSTM': models.LSTM,
    'BiLSTM': models.BiLSTM
}

con = config.Config(args)
con.set_seed(args.seed)
con.set_max_epoch(300)
con.load_train_data()
con.load_test_data()
# con.set_train_model()
con.train(model[args.model_name], args.save_name)
