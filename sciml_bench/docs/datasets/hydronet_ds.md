# Hydronet dataset

The hdf5 files include the atomic positions (data.pos), atomic numbers (data.z), one-hot encoding of atomic numbers (data.x), number of atoms in sample (data.size), index in database (data.idx), name(s) of attribute(s) to be predicted (data.name), and attribute(s) to be predicted (data.y). Various databases were created for a different studies, these are: 
1. min_data.hdf5 (6.2GBytes) - water cluster minima, where clusters contain between 3 and 30 water molecules. 
2. nonmin_data.hdf5 (1.1GBytes) - water cluster non-minima, where clusters contain between 3 and 25 water molecules. 
3. mctbp_data.hdf5 (112MBytes) - water cluster structures after a Monte Carlo Temperature Basin Paving move, where clusters contain between 3 and 30 water molecules. 
4. qm9.hdf5 (70Mbytes) – for testing
5. min.hdf5 (5.7GBytes) - water cluster minima, where clusters contain between 3 and 25 water molecules.

QM9 data from: http://quantum-machine.org/datasets/ 

Water data from: https://sites.uw.edu/wdbase/database-of-water-clusters/

#Parameters of input dataset

<HDF5 dataset "pos": shape (133885, 30, 3), type "<f4">  (n_samples, max_n_atoms, xyz coordinates)

<HDF5 dataset "size": shape (133885, 1), type "|u1"> (n_samples, n_atoms)

<HDF5 dataset "x": shape (133885, 30, 5), type "|u1"> (n_samples, max_n_atoms, one-hot encoding of z [number refers to the number of atom types present in dataset])

<HDF5 dataset "y": shape (133885, 1), type "<f4"> (n_samples, property [energy])

<HDF5 dataset "z": shape (133885, 30), type "|u1"> (n_samples, max_n_atoms [atomic number])
