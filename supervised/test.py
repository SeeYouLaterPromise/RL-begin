import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.config_game import SUPERVISED_DIR
from utils.metric_plot import plot_loss
print(os.path.abspath(os.path.join(SUPERVISED_DIR, "result/12-15-22/all_training_logs.csv")))


train_df = pd.read_csv(os.path.abspath(os.path.join(SUPERVISED_DIR, "result/12-15-22/all_training_logs.csv")))
test_df = pd.read_csv(os.path.abspath(os.path.join(SUPERVISED_DIR, "result/12-15-22/all_test_logs.csv")))
save_dir = os.path.abspath(os.path.join(SUPERVISED_DIR, "result/12-15-22"))
print(train_df)
addtional_ls = ['BC Loss', 'Penalty Loss']
plot_loss(train_df, test_df, save_dir, addtional_ls, addtional_ls)