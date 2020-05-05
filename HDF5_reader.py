import h5py
filename = "C:/Users/Andreas/Documents/GitRepositories/DenseDepth/scenenn_seg_76/scenenn_seg/scenenn_coords_005.hdf5"

with h5py.File(filename, "r") as f:
    # List all groups
    print("Keys: %s" % f.keys())
    a_group_key = list(f.keys())[0]

    # Get the data
    data = list(f[a_group_key])
    print(data[100].shape)