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
    alpha = float(sys.argv[1])
    uplo = sys.argv[2]
    x = load_tensor(curr_dir + "/float_input_0.bin")
    A = load_tensor(curr_dir + "/float_input_1.bin")
    n = int(np.sqrt(A.size))
    A = A.reshape(n, n)
    update = A + alpha * np.outer(x, x)
    golden = A.copy()
    if uplo == "lower":
        mask = np.tril(np.ones((n, n), dtype=bool))
    else:
        mask = np.triu(np.ones((n, n), dtype=bool))
    golden[mask] = update[mask]
    golden.astype(np.float32).tofile(f"float_golden_t_ssyr_0.bin")
