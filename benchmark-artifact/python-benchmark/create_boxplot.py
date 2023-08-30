import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

data_dir = Path(__file__).parent.parent.parent.resolve() / "benchmark-artifact/data"
csv_file_labels = "" # File with marked planes
def createBoxplot():
    data = np.genfromtxt(Path(data_dir) / Path('process_sequence_50_snapshot.csv'), delimiter=',')

    sns.boxplot(data=[data])
    plt.xticks([0], ['stable'])
    plt.ylabel("Time (ms.)")
    plt.savefig("boxplot.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    createBoxplot()