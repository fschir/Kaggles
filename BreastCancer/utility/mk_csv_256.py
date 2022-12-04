import pandas as pd
from pathlib import Path


def main():
    datapath = Path('../data/256')
    train_csv = datapath / 'train.csv'
    full_df = pd.read_csv(train_csv)

    # Dataset for positive and negative samples
    cancer_df = full_df[full_df['cancer'] == 1]
    clean_df = full_df[full_df['cancer'] == 0]
    print(len(cancer_df), len(clean_df))


    # Make tiny dataset for fast debugging
    debug_df = pd.concat([cancer_df[0:100], clean_df[0:100]])
    debug_df = debug_df.sample(frac=1)
    debug_df.to_csv(datapath / 'debug.csv')

    # Make balanced dataset for training
    balanced_df = pd.concat([cancer_df, clean_df.sample(len(cancer_df))])
    balanced_df = balanced_df.sample(frac=1)
    balanced_df.to_csv(datapath / 'balanced.csv')


if __name__ == "__main__":
    main()
