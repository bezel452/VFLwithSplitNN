import matplotlib.pyplot as plt
import csv
import time
import argparse

class drawFigure:
    def __init__(self, num_client, csv_path, epochs, dataset):
        self.num = num_client
        self.csv = csv_path
        self.x = [_ for _ in range(1, epochs + 1)]
        self.d = dataset
    
    def drawLossFigure(self):
        with open(self.csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                V_data = row
        V_data = list(map(float, V_data))
        t = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        filename = self.d + '_' + str(self.num) + '_'+ str(len(self.x)) + '_' + t + '.png'
        plt.plot(self.x, V_data)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Figure of Loss')
        plt.savefig(filename)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='It is the simulation of VFL with SplitNN')
    parser.add_argument('-n', '--num_client', default=2, type=int, help='The number of clients in the Simulation')
    parser.add_argument('-d', '--dataset', default='cifar10', type=str, help='The dataset used in the Simulation')
    parser.add_argument('-e', '--epochs', default=20, type=int, help='The epochs of training')
    parser.add_argument('-f', '--file_path', default=None, type=str, help='The path of csv file')
    args = parser.parse_args()
    draw = drawFigure(args.num_client, args.file_path, args.epochs, args.dataset)
    draw.drawLossFigure()
        
