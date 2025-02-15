from scripts.dbua import dbua
from utils.data import*

if __name__ == "__main__":
    # run_list = ["inclusion_layer", "inclusion", "four_layer", "two_layer",
    #     "checker2", "checker8", "1420", "1465", "1480", "1510", "1540", "1555", "1570"]
    run_list = ["two_layer"]

    # Run all examples
    for sample in run_list:
        print(sample)
        c = CTRUE[sample]
        dbua(sample, LOSS, c)