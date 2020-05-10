import argparse
import pandas as pd

def main(args):
    data = pd.read_csv(args.csvpath,dtype="string")
    data = data.astype(str)
    for index, row in data.iterrows():
        print("<DOC>")
        print("<DOCNO>" + row['cord_uid'] + "</DOCNO>")
        print("<TITLE>" + row['title'] + "</TITLE>")
        print("<TEXT>" + row['abstract']+ "</TEXT>")
        print("</DOC>")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csvpath', default="./data/metadata.csv")
    args = parser.parse_args()
    main(args)
