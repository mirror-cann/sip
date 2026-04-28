import sys
import os
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))

def load_and_gen_golden_data(filename):
    with open(filename, 'rb') as f:
        dtype_int = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dtype_map = {0: np.float32, 1: np.float32, 2: np.float16, 3: np.int32, 4: np.int64, 16: np.complex64, 6: np.uint8}
        np_type = dtype_map.get(dtype_int, np.float32)
        dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dims = np.frombuffer(f.read(8 * dim_count), dtype=np.int64)
        data = np.frombuffer(f.read(), dtype=np_type)
        data.reshape(dims)

        # scopy: golden is identical to input x (input_0)
        golden = data.astype(np_type)
        golden.astype(np_type).tofile(f"{np_type}_golden_t_scopy_{0}.bin")

if __name__ == "__main__":
    filename = curr_dir + "/float_input_0.bin"
    load_and_gen_golden_data(filename)
