import struct
from pathlib import Path

ROOT = Path("data/sift_download/sift")
OUT = Path("data/sift_download")

DIM = 128
BASE_COUNT = 100_000
QUERY_COUNT = 200

def copy_fvecs(src, dst, count):
    with open(src, "rb") as f:
        for i in range(count):
            raw_dim = f.read(4)
            if len(raw_dim) != 4:
                raise ValueError(f"Unexpected EOF at vector {i}")

            d = struct.unpack("<i", raw_dim)[0]
            if d != DIM:
                raise ValueError(f"Expected {DIM}, got {d}")

            vector = f.read(DIM * 4)
            if len(vector) != DIM * 4:
                raise ValueError(f"Incomplete vector {i}")

            dst.write(vector)

output = OUT / "sift_100k.data_new"

print("Converting SIFT...")

with open(output, "wb") as f:
    f.write(struct.pack("<iii", 4, QUERY_COUNT + BASE_COUNT, DIM))

    print("Writing 200 official queries...")
    copy_fvecs(ROOT / "sift_query.fvecs", f, QUERY_COUNT)

    print("Writing 100,000 base vectors...")
    copy_fvecs(ROOT / "sift_base.fvecs", f, BASE_COUNT)

print("\nConversion complete.")
print(f"File: {output}")
print(f"Total vectors: {QUERY_COUNT + BASE_COUNT}")
print(f"Queries: {QUERY_COUNT}")
print(f"Indexed vectors: {BASE_COUNT}")
print(f"Dimensions: {DIM}")
print(f"Size: {output.stat().st_size / (1024**2):.2f} MiB")
