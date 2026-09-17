"""Convert AASPI VDS attribute volumes (one 2D line) to float32 .npy arrays of shape (CDPs, samples)."""
import sys
import numpy as np
import openvds

def main(src, dst):
    h = openvds.open(src, ""); lay = openvds.getLayout(h); acc = openvds.getAccessManager(h)
    shape = [lay.getAxisDescriptor(i).getNumSamples() for i in range(lay.getDimensionality())]
    req = acc.requestVolumeSubset(np.zeros(6, np.int32), np.array(tuple(shape) + (1,) * (6 - len(shape)), np.int32),
                                  format=openvds.VolumeDataChannelDescriptor.Format.Format_R32)
    d = np.asarray(req.data).reshape(shape[::-1])[0]
    np.save(dst, d.astype(np.float32)); print(src, "->", dst, d.shape)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
