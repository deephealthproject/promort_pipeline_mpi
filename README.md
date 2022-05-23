# SGD MPI

This repo contains a basic MPI implementation of the EDDL SGD optimizer, along
with our use case pipeline.
 
The code requires a CUDA aware OpenMPI and UCX installation to exploit
GPUDirect features for fast MPI communications among GPUs (running both on the
same host or on different hosts). UCX allows for transparent use of NVLink,
Cuda IPC, InfiniBand and TCP.  A working environment is provided by the
*Dockerfile* in the parent folder, which automatically installs the latest
version of [**gdrcopy**](https://github.com/NVIDIA/gdrcopy).

(Note that it is also necessary to build and load the **gdrdrv** kernel module
in the host before running the container. Please note that the versions of
**gdrcopy** on the docker image and the host must be the same.)

The **code** folder has the following sub-directories:
 * **opt_mpi**: The *cpp* sub-folder includes the code to implement mpi
   functionalities along with the extension of the SGD optimizer. The *pybind*
   sub-folder includes the code to create python bindings.
 * **promort-distributed**: Code of the distributed training of our use case.
 * **k8s**: Kubernetes configuration files used in our tests
 * **varia**: Auxiliary files used during the Docker build.  
