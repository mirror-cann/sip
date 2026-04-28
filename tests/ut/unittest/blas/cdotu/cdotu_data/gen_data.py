import sys
import os
import numpy as np

curr_dir = os.path.dirname(os.path.realpath(__file__))

def load_tensor(filename):
    with open(filename, 'rb') as f:
        dtype_int = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dtype_map = {0: np.float32, 1: np.float32, 2: np.float16, 3: np.int32,
                     4: np.int64, 16: np.complex64, 6: np.uint8}
        np_type = dtype_map.get(dtype_int, np.float32)
        dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]
        f.read(8 * dim_count)
        data = np.frombuffer(f.read(), dtype=np_type)
        return np_type, data

if __name__ == "__main__":
    np_type, x = load_tensor(curr_dir + "/complex64_input_0.bin")
    _, y = load_tensor(curr_dir + "/complex64_input_1.bin")
    golden = np.dot(x, y)
    np.array(golden, dtype=np_type).tofile(f"{np_type}_golden_t_cdotu_0.bin")
