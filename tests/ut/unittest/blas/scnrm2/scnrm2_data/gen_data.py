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
        return data

if __name__ == "__main__":
    x = load_tensor(curr_dir + "/complex64_input_0.bin")
    golden = np.linalg.norm(x)
    np.array(golden, dtype=np.float32).tofile(f"float_golden_t_scnrm2_0.bin")
