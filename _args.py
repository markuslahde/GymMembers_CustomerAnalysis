import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Parse model training options')
    
    parser.add_argument('-data_dir', '--data_dir', default="data",
                        help='Directory for production data')
    
    parser.add_argument('-production_data', '--production_data', default='production_data.csv',
                        help='Production data file')
    
    parser.add_argument('-visualizations_dir', '--visualizations_dir', default="visualizations",
                        help='Directory for visualizations')
    
    parser.add_argument("--palette", type=str, default="gist_ncar", 
                    choices=["deep", "muted", "pastel", "bright", "dark", "colorblind", "coolwarm", "viridis"],
                    help="Seaborn color palette to use for plots.")
    
    parser.add_argument('-style', '--style', default="white", 
                    choices=["whitegrid", "darkgrid", "white", "dark", "ticks"],
                    help="Seaborn style to use for plots.")

    parser.add_argument('-font', '--font', default="Sans", 
                    #choices=["Arial", "Sans", "Serif", "Times New Roman", "Comic Sans MS"],
                    help="Font to use for plot text.")

    parser.add_argument('-context', '--context', default="talk", 
                    choices=["paper", "notebook", "talk", "poster"],
                    help="Seaborn context to use for plots.")

    args = parser.parse_args()
    return args
