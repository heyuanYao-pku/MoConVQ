from mpi4py import MPI
import numpy as np 
mpi_comm = MPI.COMM_WORLD
mpi_world_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()
def gather_dict_ndarray(dict):
    info = {key: [list(dict[key].shape), dict[key].dtype] for key in dict.keys()}
    info = mpi_comm.bcast(info, root = mpi_world_size - 1)
    
    res = {}
    for key, value in info.items():
        shape, dtype = value
        send_buf = dict[key] if key in dict else MPI.IN_PLACE
        length = 0 if send_buf is MPI.IN_PLACE else np.prod(send_buf.shape)
        length_list = mpi_comm.allreduce([length])
        shape[0] = sum(length_list)//int(np.prod(shape[1:]))
        recv_buf = np.zeros(shape, dtype = dtype) if mpi_rank == 0 else None
        mpi_comm.Gatherv(send_buf, (recv_buf, length_list), root = 0) 
        res[key] = recv_buf
    return res