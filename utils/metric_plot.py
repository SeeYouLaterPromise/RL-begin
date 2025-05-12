import matplotlib.pyplot as plt

""" train_df: 'Epoch', 'Train Loss', 'Train Accuracy'  """
""" test_df:  'Epoch', 'Test Loss', 'Test Accuracy',  
              'Precision', 'Precision Std', 'Recall', 
              'Recall Std', 'F1 Score', 'F1 Score Std' """

def plot_loss_helper(df, plot_ls):
    for plot_cell in plot_ls:
        if plot_cell in df.columns:
            plt.plot(df['Epoch'], df[plot_cell], label=plot_cell)
        else:
            print(f"[Warning] '{plot_cell}' not found in DataFrame.")

def plot_loss(train_df, test_df, save_dir, train_plot_ls=['Train Loss'], test_plot_ls=['Test Loss'], name=None):
    import os
    plt.figure(figsize=(10, 6))

    plot_loss_helper(train_df, train_plot_ls)
    if test_df is not None:
        plot_loss_helper(test_df, test_plot_ls)

    name = f"{name}_" if name else ""
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    save_path = os.path.join(save_dir, f"{name}loss_plot.png")
    plt.savefig(save_path)
    plt.close()


def plot_acc(train_df, test_df, save_dir, name=None):
    if not name:
        name = ""
    else:
        name = name + "_"

    plt.figure(figsize=(10, 6))
    plt.plot(train_df['Epoch'], train_df['Train Accuracy'], label='Train Accuracy')
    plt.plot(test_df['Epoch'], test_df['Test Accuracy'], label='Test Accuracy')
    plt.fill_between(test_df['Epoch'],
                     test_df['Test Accuracy'] - test_df['Test Accuracy Std'],
                     test_df['Test Accuracy'] + test_df['Test Accuracy Std'],
                     alpha=0.2, label='Accuracy Std')

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy')
    plt.legend()
    plt.savefig(f'{save_dir}/{name}accuracy_plot.png')
    plt.close()


def plot_classify_metrics(test_df, save_dir, name=None):
    """Generate and save plots for training and test metrics."""

    if not name:
        name = ""
    else:
        name = name + "_"


    # Plot Precision, Recall, and F1 Score
    plt.figure(figsize=(10, 6))
    plt.plot(test_df['Epoch'], test_df['Precision'], label='Precision')
    plt.plot(test_df['Epoch'], test_df['Recall'], label='Recall')
    plt.plot(test_df['Epoch'], test_df['F1 Score'], label='F1 Score')
    plt.fill_between(test_df['Epoch'],
                     test_df['Precision'] - test_df['Precision Std'],
                     test_df['Precision'] + test_df['Precision Std'],
                     alpha=0.2, label='Precision Std')
    plt.fill_between(test_df['Epoch'],
                     test_df['Recall'] - test_df['Recall Std'],
                     test_df['Recall'] + test_df['Recall Std'],
                     alpha=0.2, label='Recall Std')
    plt.fill_between(test_df['Epoch'],
                     test_df['F1 Score'] - test_df['F1 Score Std'],
                     test_df['F1 Score'] + test_df['F1 Score Std'],
                     alpha=0.2, label='F1 Score Std')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Test Precision, Recall, and F1 Score')
    plt.legend()
    plt.savefig(f'{save_dir}/{name}metrics_plot.png')
    plt.close()