# BPFL: Towards Efficient Byzantine-Robust and Provably Privacy-Preserving Federated Learning

## Dependencies
In BPFL, the valid check part is coded by C++ with [libsnark](https://github.com/scipr-lab/libsnark), and the FL part is coded by Python, so you should first deploy libsnark and [pybind11](https://github.com/pybind/pybind11), which is a lightweight header-only library that mainly to create Python bindings of existing C++ code, to your machine before running. For deployment details, please refer to [libsnark](https://github.com/scipr-lab/libsnark) and [pybind11](https://github.com/pybind/pybind11).
## Requirements
- pytorch
- CUDA
- numpy
- torchvision
- [phe](https://github.com/data61/python-paillier)
## Run
1. You need to first compile the [main.cpp](https://github.com/BPFL/BPFL/blob/main/ZKP/src/main.cpp) file in the [ZKP](https://github.com/BPFL/BPFL/tree/main/ZKP/src) directory to obtain a dynamic link library named xxx.so and rename it as ZKP.so.
