import torch
from torch.nn import functional as F
import numpy as np
import os.path as osp
from typing import Optional, Callable
import gdown
import sys
import tempfile
import os
from copy import deepcopy as copy
from pathlib import Path
from torch_geometric.data import DataListLoader, DataLoader, InMemoryDataset, Data, extract_zip, download_url
import h5py
import logging
import requests
import warnings
import zipfile
import re
import tarfile
import tempfile
from urllib import request as request
from ase.io.extxyz import read_xyz
from ase.units import Debye, Bohr, Hartree, eV
from ase.db import connect
import shutil
# import fair_research_login

class PrepackedDataset(torch.utils.data.Dataset): #InMemoryDataset:
    def __init__(self, loader_list, split_file, dataset_type, 
                 max_num_atoms=None, num_elements=None,
                 shuffle=True, mode="train", directory="./data/cached_dataset/"):
        
        self.dataset = []
        self.shuffle = shuffle
        self.directory = directory
        self.mode = mode
        self.split_file = split_file
        self.dataset_type = dataset_type
        self.max_num_atoms = max_num_atoms
        self.num_elements = num_elements

        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Create .hdf5 from .pt
        if loader_list is not None:           
            for i in range(len(loader_list)): 
                loader = loader_list[i]
                self.create_container(loader)
                logging.info("Finishing processing data...")
                for index, data in enumerate(loader):
                    self.x[index][:data.size, :] = copy(data.x).to(torch.uint8)
                    self.z[index][:data.size] = copy(data.z).to(torch.uint8)
                    self.pos[index][:data.size] = copy(data.pos).to(torch.float32)
                    self.y[index] = copy(data.y).to(torch.float32)
                    self.size[index] = copy(data.size).to(torch.uint8)
                self.save_data()
                
    def create_container(self, loader):
        tmp = next(iter(loader))
        n_elements = len(loader)
        
        self.x = np.zeros((n_elements, self.max_num_atoms, self.num_elements), dtype=np.uint8)
        self.z = np.zeros((n_elements, self.max_num_atoms), dtype=np.uint8)
        self.pos = np.zeros((n_elements, self.max_num_atoms, 3))
        self.y = np.zeros((n_elements, 1))
        self.size = np.zeros((n_elements, 1), dtype=np.uint8)
        
    def save_data(self):
        logging.info("Saving cached data in disk...")
        dataset = h5py.File(os.path.join(self.directory,f"{self.dataset_type}.hdf5"), "w")
        dataset.create_dataset("z", dtype=np.uint8, data=self.z)
        dataset.create_dataset("x", dtype=np.uint8, data=self.x)
        dataset.create_dataset("pos", dtype=np.float32, data=self.pos)
        dataset.create_dataset("y", dtype=np.float32, data=self.y)
        dataset.create_dataset("size", dtype=np.uint8, data=self.size)
        dataset.close()
        
    def load_data(self, idx_type):
        # logging.info("Loading cached data from disk...")
        print("Loading cached data from disk...\n")
        dataset = h5py.File(os.path.join(self.directory, f"{self.dataset_type}.hdf5"), "r")

        S = np.load(self.split_file)
        self.mode_idx = S[f'{idx_type}_idx']

        data_list = []
        for i in range(len(self.mode_idx)):
            index = self.mode_idx[i]
            cluster_size = dataset["size"][index][0]
            
            z = torch.from_numpy(dataset["z"][index][:cluster_size])
            x = torch.from_numpy(dataset["x"][index][:cluster_size])
            pos = torch.from_numpy(dataset["pos"][index][:cluster_size])
            # pos.requires_grad = True
            y = torch.from_numpy(dataset["y"][index])
            size = torch.from_numpy(dataset["size"][index])
            data = Data(x=x, z=z, pos=pos, y=y, size=size)
            data_list.append(data)
        print("Data list size: {:5d} \n".format(len(data_list)), flush=True)
        # return self.collate(data_list)
        return data_list
    def __len__(self):
        return len(self.z)

    def __getitem__(self, index):
        return self.x[index], self.z[index], self.pos[index], self.y[index], self.f[index], self.size[index]


    
class WaterMinimaDataSet(InMemoryDataset):
    def __init__(self,
                 sample,
                 root: Optional[str] = './data/cached_dir',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None
                 ):
        """

        Args:
            sample: Dataset name (no extension)
            root: Directory where processed output will be saved
            transform: Transform to apply to data
            pre_transform: Pre-transform to apply to data
            pre_filter: Pre-filter to apply to data
        """
        self.atom_types = [1, 8]
        self.max_num_atoms = 90
        
        self.root = root
        self.sample = sample
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        # NB: database file
        return f'{self.sample}.db'
    
    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return self.root

    @property
    def processed_file_names(self):
        return f'{self.sample}.pt'

    def download(self):
        """
        The base class will automatically look for a file that matches the raw_file_names property in a directory named 'raw'. If it doesn't find it, it will download the data using this method
        :return:
        """

        logging.info("Downloading HydroNet water cluster data...")
        
        # ID for PNNL DataHub collection
        collection = 'f58973c0-08c1-43a7-9a0e-71f54ddc973c'
        zip_file_name = 'W3-W30_all_geoms_TTM2.1-F.zip'

        # Guest Collections only require the https scope.
        scopes = [f'https://auth.globus.org/scopes/{collection}/https']

        # Fetch an HTTPS token
        client = fair_research_login.NativeClient(client_id='7414f0b4-7d05-4bb6-bb00-076fa3f17cf5')
        tokens = client.login(requested_scopes=scopes)
        # User is given URL to visit, leads to Globus login via institution, Google, or ORCID, returns a token

        https_token = tokens[collection]['access_token']

        
        # Fetch the file
        raw_url = 'https://g-83fdd0.1beed.03c0.data.globus.org/static_datasets/W3-W30_all_geoms_TTM2.1-F.zip'
        filename = Path(raw_url).name
        headers = {'Authorization': f'Bearer {https_token}', 'Content-Disposition': 'download'}
        with requests.get(raw_url, stream=True, headers=headers) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=4096): 
                    f.write(chunk)
                    
        logging.info("Extracting data...")            
        with zipfile.ZipFile(zip_file_name) as zip_ref:
            zip_ref.extractall('./')
            
        os.remove(zip_file_name)
        shutil.move(osp.join('./water_db','ALL_geoms_all_sorted.db'), osp.join(self.root, f'{self.sample}.db'))
        shutil.rmtree('./water_db')

        logging.info("Done downloading data.")

    def process(self):
        """
        Processes the raw data and saves it as a Torch Geometric data set

        The steps does all pre-processing required to put the data extracted from the database into graph 'format'. Several transforms are done on the data in order to generate the graph structures used by training.

        The processed dataset is automatically placed in a directory named processed, with the name of the processed file name property. If the processed file already exists in the correct directory, the processing step will be skipped.

        :return: Torch Geometric Dataset
        """
        # NB: coding for atom types
        types = {'H': 0, 'O': 1}

        data_list = []
        
        dbfile = osp.join(self.root, self.raw_file_names)
        assert osp.isfile(dbfile), f"Database file not found: {dbfile}"

        with connect(dbfile) as conn:
            center = True
            for i in range(len(conn)):
                row = conn.get(id=i+1)
                name = ['energy']
                mol = row.toatoms()
                y = torch.tensor(row.data[name[0]], dtype=torch.float) #potential energy
                if center:
                    pos = mol.get_positions() - mol.get_center_of_mass()
                else:
                    pos = mol.get_positions()
                pos = torch.tensor(mol.get_positions(), dtype=torch.float) #coordinates
                size = int(pos.size(dim=0))
                type_idx = [types.get(i) for i in mol.get_chemical_symbols()]
                atomic_number = mol.get_atomic_numbers()
                z = torch.tensor(atomic_number, dtype=torch.long)
                x = F.one_hot(torch.tensor(type_idx, dtype=torch.long),
                              num_classes=len(types))

                data = Data(x=x, z=z, pos=pos, y=y, size=size, name=name, idx=i)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                    # The graph edge_attr and edge_indices are created when the transforms are applied
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                        
                data_list.append(data)
                    
        torch.save(self.collate(data_list), self.processed_paths[0])
        return self.collate(data_list)
    
    
    
class QM9DataSet(InMemoryDataset):
    
    def __init__(self,
             sample,
             root: Optional[str] = None,
             transform: Optional[Callable] = None,
             pre_transform: Optional[Callable] = None,
             pre_filter: Optional[Callable] = None
             ):
        """

        Args:
            sample: Dataset name (no extension)
            root: Directory where processed output will be saved
            transform: Transform to apply to data
            pre_transform: Pre-transform to apply to data
            pre_filter: Pre-filter to apply to data
        """
        
        # properties
        self.atom_types = [1,6,7,8,9]
        self.max_num_atoms = 30
        
        A = "rotational_constant_A"
        B = "rotational_constant_B"
        C = "rotational_constant_C"
        mu = "dipole_moment"
        alpha = "isotropic_polarizability"
        homo = "homo"
        lumo = "lumo"
        gap = "gap"
        r2 = "electronic_spatial_extent"
        zpve = "zpve"
        U0 = "energy_U0"
        U = "energy_U"
        H = "enthalpy_H"
        G = "free_energy"
        Cv = "heat_capacity"

        self.available_properties = [
            A,
            B,
            C,
            mu,
            alpha,
            homo,
            lumo,
            gap,
            r2,
            zpve,
            U0,
            U,
            H,
            G,
            Cv,
        ]

        #self.reference = {zpve: 0, U0: 1, U: 2, H: 3, G: 4, Cv: 5}

        self.units = dict(
            zip(
                self.available_properties,
                [
                    1.0,
                    1.0,
                    1.0,
                    Debye,
                    Bohr ** 3,
                    Hartree,
                    Hartree,
                    Hartree,
                    Bohr ** 2,
                    Hartree,
                    Hartree,
                    Hartree,
                    Hartree,
                    Hartree,
                    1.0,
                ],
            )
        )

        self.root = root
        self.sample = sample
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        # NB: database file
        return f'{self.sample}.db'
    
    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return self.root

    @property
    def processed_file_names(self):
        return f'{self.sample}.pt'

    def download(self):
        """
        The base class will automatically look for a file that matches the raw_file_names property in a directory named 'raw'. If it doesn't find it, it will download the data using this method
        :return:
        """
        logging.info("Downloading GDB-9 data...")
        raw_url = "https://ndownloader.figshare.com/files/3195389"
        
        tmpdir = tempfile.mkdtemp("gdb9")
        tar_path = os.path.join(tmpdir, "gdb9.tar.gz")
        raw_path = os.path.join(tmpdir, "gdb9_xyz")

        request.urlretrieve(raw_url, tar_path)

        logging.info("Extracting files...")
        tar = tarfile.open(tar_path)
        tar.extractall(raw_path)
        tar.close()
        
        ordered_files = sorted(
            os.listdir(raw_path), key=lambda x: (int(re.sub("\D", "", x)), x)
        )

        all_atoms = []
        all_properties = []

        logging.info("Writing to database...")
        for i in range(len(ordered_files)):
            xyzfile = os.path.join(raw_path, ordered_files[i])

            if (i + 1) % 10000 == 0:
                logging.info("Parsed: {:6d} / 133885".format(i + 1))
            properties = {}
            tmp = os.path.join(tmpdir, "tmp.xyz")

            with open(xyzfile, "r") as f:
                lines = f.readlines()
                l = lines[1].split()[2:]
                for pn, p in zip(self.available_properties, l):
                    properties[pn] = np.array([float(p) * self.units[pn]])
                with open(tmp, "wt") as fout:
                    for line in lines:
                        fout.write(line.replace("*^", "e"))

            with open(tmp, "r") as f:
                ats = list(read_xyz(f, 0))[0]
            all_atoms.append(ats)
            all_properties.append(properties)
            
        dbpath = osp.join(self.root,self.raw_file_names)

        with connect(dbpath) as conn:
            for at, prop in zip(all_atoms, all_properties):
                conn.write(at, data=prop)

        shutil.rmtree(tmpdir)
        logging.info("Done downloading data.")
        
    def process(self):
        """
        Processes the raw data and saves it as a Torch Geometric data set

        The steps does all pre-processing required to put the data extracted from the database into graph 'format'. Several transforms are done on the data in order to generate the graph structures used by training.

        The processed dataset is automatically placed in a directory named processed, with the name of the processed file name property. If the processed file already exists in the correct directory, the processing step will be skipped.

        :return: Torch Geometric Dataset
        """
        # NB: coding for atom types
        types = {'H':0,'C':1,'N':2,'O':3,'F':4}

        data_list = []
        
        dbfile = osp.join(self.root, self.raw_file_names)
        assert osp.isfile(dbfile), f"Database file not found: {dbfile}"

        with connect(dbfile) as conn:
            center = True
            for i in range(len(conn)):
                row = conn.get(id=i+1)
                name = ['energy_U']
                mol = row.toatoms()
                y = torch.tensor(row.data[name[0]], dtype=torch.float) #potential energy
                if center:
                    pos = mol.get_positions() - mol.get_center_of_mass()
                else:
                    pos = mol.get_positions()
                pos = torch.tensor(mol.get_positions(), dtype=torch.float) #coordinates
                size = int(pos.size(dim=0))
                type_idx = [types.get(i) for i in mol.get_chemical_symbols()]
                atomic_number = mol.get_atomic_numbers()
                z = torch.tensor(atomic_number, dtype=torch.long)
                x = F.one_hot(torch.tensor(type_idx, dtype=torch.long),
                              num_classes=len(types))

                data = Data(x=x, z=z, pos=pos, y=y, size=size, name=name, idx=i)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                    # The graph edge_attr and edge_indices are created when the transforms are applied
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                        
                data_list.append(data)
                    
        torch.save(self.collate(data_list), self.processed_paths[0])
        return self.collate(data_list)

    
class PCQM4Mv2DataSet(InMemoryDataset):
    
    def __init__(self,
             sample,
             root: Optional[str] = None,
             transform: Optional[Callable] = None,
             pre_transform: Optional[Callable] = None,
             pre_filter: Optional[Callable] = None
             ):
        """

        Args:
            sample: Dataset name (no extension)
            root: Directory where processed output will be saved
            transform: Transform to apply to data
            pre_transform: Pre-transform to apply to data
            pre_filter: Pre-filter to apply to data
        """
        
        self.atom_types = list(range(1,36))
        self.max_num_atoms = 53
        
        self.root = root
        self.sample = sample
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        # NB: database file
        dbfiles = [x for x in os.listdir(self.raw_dir) if x.endswith('db')]
        return dbfiles
    
    @property
    def raw_dir(self):
        return osp.join(self.root,'noopt_dbs')

    @property
    def processed_dir(self):
        return self.root

    @property
    def processed_file_names(self):
        return f'{self.sample}.pt'

    def download(self):
        """
        The base class will automatically look for a file that matches the raw_file_names property in a directory named 'raw'. If it doesn't find it, it will download the data using this method
        :return:
        """
        print('No download available.')
        
    def process(self):
        """
        Processes the raw data and saves it as a Torch Geometric data set

        The steps does all pre-processing required to put the data extracted from the database into graph 'format'. Several transforms are done on the data in order to generate the graph structures used by training.

        The processed dataset is automatically placed in a directory named processed, with the name of the processed file name property. If the processed file already exists in the correct directory, the processing step will be skipped.

        :return: Torch Geometric Dataset
        """
        # NB: coding for atom types
        types = {'H':0, 'He':1, 'Li':2, 'Be':3, 'B':4, 'C':5, 'N':6, 'O':7, 
                 'F':8, 'Ne':9, 'Na':10, 'Mg':11, 'Al':12, 'Si':13, 'P':14, 
                 'S':15, 'Cl':16, 'Ar':17, 'K':18, 'Ca':19, 'Sc':20, 'Ti':21, 
                 'V':22, 'Cr':23, 'Mn':24, 'Fe':25, 'Co':26, 'Ni':27, 'Cu':28, 
                 'Zn':29, 'Ga':30, 'Ge':31, 'As':32, 'Se':33,'Br':34}

        data_list = []
        #print(self.raw_dir,self.raw_file_names) 
        for dbfile in self.raw_file_names:
            with connect(osp.join(self.raw_dir, dbfile)) as conn:
                center = True
                for i in range(len(conn)):
                    row = conn.get(id=i+1)
                    name = ['y']   #calculated HOMO-LUMO gap
                    mol = row.toatoms()
                    y = torch.tensor(row.data[name[0]], dtype=torch.float) 
                    if center:
                        pos = mol.get_positions() - mol.get_center_of_mass()
                    else:
                        pos = mol.get_positions()
                    pos = torch.tensor(mol.get_positions(), dtype=torch.float) #coordinates
                    size = int(pos.size(dim=0))
                    type_idx = [types.get(i) for i in mol.get_chemical_symbols()]
                    atomic_number = mol.get_atomic_numbers()
                    z = torch.tensor(atomic_number, dtype=torch.long)
                    x = F.one_hot(torch.tensor(type_idx, dtype=torch.long),
                                      num_classes=len(types))    

                    data = Data(x=x, z=z, pos=pos, y=y, size=size, name=name, idx=i)

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                        # The graph edge_attr and edge_indices are created when the transforms are applied
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)

                    data_list.append(data)
                    
        torch.save(self.collate(data_list), self.processed_paths[0])
        return self.collate(data_list)
