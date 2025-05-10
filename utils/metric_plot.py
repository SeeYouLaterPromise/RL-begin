import matplotlib.pyplot as plt

""" train_df: 'Epoch', 'Train Loss', 'Train Accuracy'  """
""" test_df:  'Epoch', 'Test Loss', 'Test Accuracy',  
              'Precision', 'Precision Std', 'Recall', 
              'Recall Std', 'F1 Score', 'F1 Score Std' """

def plot_loss(train_df, test_df, save_dir, name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_df['Epoch'], train_df['Train Loss'], label='Train Loss')

    if test_df is not None:
        plt.plot(test_df['Epoch'], test_df['Test Loss'], label='Test Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(f'{save_dir}/{name}_loss_plot.png')
    plt.close()

def plot_acc(train_df, test_df, save_dir, name):
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
    plt.savefig(f'{save_dir}/{name}_accuracy_plot.png')
    plt.close()


def plot_classify_metrics(test_df, save_dir, name):
    """Generate and save plots for training and test metrics."""

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
    plt.savefig(f'{save_dir}/{name}_metrics_plot.png')
    plt.close()