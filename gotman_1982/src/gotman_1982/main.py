import numpy as np
from epilepsy2bids.annotations import Annotations
from epilepsy2bids.eeg import Eeg

from gotman_1982.gotman import gotman_algorithm

def main(edf_file, outFile):
    
    eeg = Eeg.loadEdfAutoDetectMontage(edfFile=edf_file)
    if eeg.montage is Eeg.Montage.UNIPOLAR:
        eeg.reReferenceToBipolar()

    t = np.arange(0, eeg.data.shape[1]) / eeg.fs
    hypMask = gotman_algorithm(t, eeg.data, eeg.fs)
    hyp = Annotations.loadMask(hypMask, eeg.fs)
    hyp.saveTsv(outFile)