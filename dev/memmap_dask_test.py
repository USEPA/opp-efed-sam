from distributed import Client
import numpy as np


def generate_sample(a, b, c):
    return np.ones((a, b)) * c


def modify_sample(mmap, z, val):
    """ Use dask to modify a matrix """
    batch = []
    for i in range(z):
        old_vals = array_pull(mmap, i)
        print(old_vals)
        new_vals = old_vals + val
        print(new_vals)
        batch.append(dask_client.submit(array_write, mmap, i, new_vals))
    a = dask_client.gather(batch)
    print(a)
    print(mmap[i])



def modify_sample_simple(mmap, z, val):
    """ Modify a matrix the normal way """
    for i in range(z):
        old_vals = array_pull(mmap, i)
        new_vals = old_vals + val
        mmap[i] = new_vals


def array_pull(mmap, i):
    return mmap[i]


def array_write(mmap, i, array):
    print(f"writing {array} to pos {i}")
    mmap[i, :] = array
    mmap.flush()


def build_test_matrix(x, y, z):
    """ Use dask to construct a matrix """
    batch = []
    for i in range(z):
        batch.append(dask_client.submit(generate_sample, x, y, i))
    return np.array(dask_client.gather(batch))


def sampling_test(mmap, sequence):
    """ Use dask to grab slices in parallel """
    batch = []
    for i in sequence:
        batch.append(dask_client.submit(array_pull, mmap, i))
    return np.array(dask_client.gather(batch))


test_shape = (3, 3, 3)

# Initialize client
dask_client = Client(processes=False)

# Get the sample data
sample = build_test_matrix(*test_shape)
print("Test 1:")
print(sample)

# Initialize memmap
mm = np.memmap('test.dat', dtype=np.float32, mode='w+', shape=test_shape)
mm[:] = sample
del mm

# Grab a slice of the memmap
mm = np.memmap('test.dat', dtype=np.float32, mode='r+', shape=test_shape)
print("Test 2:")
print(mm[2])

# Get a bunch of slices
seq = [1, 1, 2, 1, 0]
result = sampling_test(mm, seq)
print("Test 3:")
print(result)

# Modify some slices
test_val = 77
# modify_sample(mm, test_shape[2], test_val)
modify_sample(mm, 1, test_val)
print("Test 4:")
for i in range(test_shape[2]):
    print(mm[i])
