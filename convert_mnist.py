import gzip
import struct
import numpy as np

source = "/home/fed/myprojects/datasets/mnist_raw/train-images-idx3-ubyte.gz"
output = "/home/fed/myprojects/datasets/mnist.data_new"

print("Reading MNIST IDX images...")

with gzip.open(source, "rb") as f:
    magic, N, rows, cols = struct.unpack(">4I", f.read(16))

    assert magic == 2051, f"Unexpected IDX magic number: {magic}"
    assert N == 60000
    assert rows == 28 and cols == 28

    X = np.frombuffer(f.read(), dtype=np.uint8)
    assert X.size == N * rows * cols

    X = X.reshape(N, rows * cols).astype(np.float32)

print("Writing DB-LSH binary dataset...")

with open(output, "wb") as f:
    f.write(struct.pack("<3i", 4, N, rows * cols))
    X.tofile(f)

print(f"Saved: {output}")
print(f"Vectors: {N}")
print(f"Dimensions: {rows * cols}")
print(f"Pixel range: {X.min()} to {X.max()}")
print("Conversion complete.")
