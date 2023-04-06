import os, subprocess, shlex, sys
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()

def nvme_staging(data_dir):
    user = os.environ.get('LSFUSER')
    nvme_dir = '/mnt/bb/%s' %(user)
    src = "%s/batch_train_%d.db" %(data_dir, comm_rank)
    #src = "%s/batch_train_0.db" %(data_dir)
    trg = "%s/batch_train_%d.db" %(nvme_dir, comm_rank)
    #cp_args = "cp -r %s/batch_%d.db %s/batch_%d.db" %(data_dir, comm_rank, nvme_dir, comm_rank)
    cp_args = "cp -r %s %s" %(src, trg)
    cp_args = shlex.split(cp_args)
    if not os.path.exists(trg):
        try:
            subprocess.run(cp_args, check=True)
        except subprocess.SubprocessError as e:
            print("rank %d: %s" % (comm_rank, format(e)))

def nvme_staging_ftf(data_dir):
    user = os.environ.get('USER')
    nvme_dir = '/mnt/bb/%s' %(user)
    src = "%s/batch_%d.tfrecords" %(data_dir, comm_rank)
    trg = "%s/batch_%d.tfrecords" %(nvme_dir, comm_rank)
    #cp_args = "cp -r %s/batch_%d.db %s/batch_%d.db" %(data_dir, comm_rank, nvme_dir, comm_rank)
    cp_args = "cp %s %s" %(src, trg)
    cp_args = shlex.split(cp_args)
    if not os.path.exists(trg):
        try:
            subprocess.run(cp_args, check=True)
        except subprocess.SubprocessError as e:
            print("rank %d: %s" % (comm_rank, format(e)))

def nvme_purging():
    user = os.environ.get('USER')
    nvme_dir = '/mnt/bb/%s' %(user)
    cp_args = "rm -r %s/batch_%d.db " % (nvme_dir, comm_rank)
    cp_args = shlex.split(cp_args)
    try:
        subprocess.run(cp_args, check=True)
    except subprocess.SubprocessError as e:
        print("rank %d: %s" % (comm_rank, format(e)))

if __name__ == "__main__":
    user = os.environ.get('LSFUSER')
    mpi_host = MPI.Get_processor_name()
    nvme_dir = '/mnt/bb/%s' % user
    if len(sys.argv) > 2:
        data_dir = sys.argv[-2]
        file_type = sys.argv[-1]
        if file_type == 'tfrecord':
            nvme_staging = nvme_staging_ftf
        #purge = bool(sys.argv[-1])
        local_rank_0 = not bool(comm_rank % 6)
        if local_rank_0:
            print('nvme contents on %s: %s '%(mpi_host,format(os.listdir(nvme_dir))))
        comm.Barrier()
        # purge
        #if purge: 
        #    nvme_purging()
        #comm.Barrier()
        #if local_rank_0: 
        #    print('nvme purged on %s' % mpi_host)
        #comm.Barrier()
        # check purge
        #if local_rank_0:
        #    print('nvme contents on %s: %s '%(mpi_host ,format(os.listdir(nvme_dir))))
        #comm.Barrier()
        # stage
        if local_rank_0 : print('begin staging on all nodes')
        nvme_staging(data_dir)
        comm.Barrier()
        if local_rank_0:
            print('all local ranks finished nvme staging on %s' % mpi_host)
        comm.Barrier()
        # check stage 
        if local_rank_0:
           print('nvme contents on %s: %s '%(mpi_host, format(os.listdir(nvme_dir))))
        comm.Barrier()
        sys.exit(0)
