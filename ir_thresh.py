#!/usr/bin/env python

import argparse

def main(args):
    with open(args.infile, 'r') as f:
        curr = ""
        top_score = 0.0
        for line in f:
            line = line.rstrip()
            col = line.split(' ')
            if col[0] != curr:
                curr = col[0]
                top_score = float(col[4])
            if float(col[4]) >= (top_score - args.threshold):
                print(line)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", type=str)
    parser.add_argument("--threshold", "-t", type=float)
    args = parser.parse_args()

    main(args)
    

