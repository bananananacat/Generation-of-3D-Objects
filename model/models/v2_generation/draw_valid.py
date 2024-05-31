import pickle
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw valid plot")
    parser.add_argument("-p", dest="path", required=True)
    
    args = parser.parse_args()
    
    with open(args.path, 'rb') as file:
        valid_dict = pickle.load(file)
    
    plt.plot(valid_dict['step'], valid_dict['valid'])
    
    plt.xlabel("step")
    plt.ylabel("f1-score")
    
    plt.show()