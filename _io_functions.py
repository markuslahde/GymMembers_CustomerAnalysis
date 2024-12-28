import os
import pandas as pd

def csv_to_df_loader(source_dir, source_file):
    file_path = os.path.join(source_dir, source_file)
    target_df = pd.read_csv(file_path)
    tags = source_dir
    return target_df, tags