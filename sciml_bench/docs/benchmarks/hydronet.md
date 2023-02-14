# Hydronet

The scientific value of SchNet model value is twofold: molecular property prediction and interatomic potential (when trained with forces).
The water cluster data is unique in standard benchmarks in that it provides long-range hydrogen bonding interactions. These hydrogen bonding interactions are what makes water such a good solvent and imparts the unique properties of water, so they are important to capture not only for modelling water itself but for modelling molecules and biomolecules solvated in water. The output is the predicted property. For qm9, this is “energy_U” (internal energy at 298.15 K). For the water dataset, this is “energy” (binding energy).The Hydronet benchmark uses the Schnet model which calculates energy and forces. There are about ~200k trainable parameters. The inference operation over the entire test set be performed by running the "test_set_errors.py" program.

# Datasets
QM9 data from: http://quantum-machine.org/datasets/ 
Water data from: https://sites.uw.edu/wdbase/database-of-water-clusters/

The hdf5 files include the atomic positions (data.pos), atomic numbers (data.z), one-hot encoding of atomic numbers (data.x), number of atoms in sample (data.size), index in database (data.idx), name(s) of attribute(s) to be predicted (data.name), and attribute(s) to be predicted (data.y). Various databases were created for a different studies, these are: 
1. min_data.hdf5 (6.2GBytes) - water cluster minima, where clusters contain between 3 and 30 water molecules. 
2. nonmin_data.hdf5 (1.1GBytes) - water cluster non-minima, where clusters contain between 3 and 25 water molecules. 
3. mctbp_data.hdf5 (112MBytes) - water cluster structures after a Monte Carlo Temperature Basin Paving move, where clusters contain between 3 and 30 water molecules. 
4. qm9.hdf5 (70Mbytes) – for testing
5. min.hdf5 (5.7GBytes) - water cluster minima, where clusters contain between 3 and 25 water molecules.

