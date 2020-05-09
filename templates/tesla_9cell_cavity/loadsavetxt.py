from optparse import OptionParser
import numpy as np

def main():

    parser = OptionParser()
    (options, args) = parser.parse_args()

    original_file = args[0]
    new_file = args[1]

    data = np.loadtxt(original_file, skiprows=1)
    np.savetxt(nes_file, data, header='R Z Er Ez Bphi', comments=' ')


if __name__ == '__main__':
    main()
