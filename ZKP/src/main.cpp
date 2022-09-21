#include <libsnark/common/default_types/r1cs_gg_ppzksnark_pp.hpp>
#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>
#include <libsnark/gadgetlib1/pb_variable.hpp>
#include <pybind11/pybind11.h>
#include "gadget.hpp"
#include <python3.8/Python.h>
#include <pybind11/stl.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <chrono>
#include <sstream>

namespace py = pybind11;
using namespace libsnark;
using namespace std;

typedef libff::Fr<default_r1cs_gg_ppzksnark_pp> FieldT;

class pack_ck {
    public:
    FieldT check_value;
    pack_ck(FieldT a){
        check_value=a;
    }
};

r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp> generate_keypairs(int dim){
    default_r1cs_gg_ppzksnark_pp::init_public_params();
    protoboard<FieldT> pb;
    valid_check<FieldT> check(pb, dim);
    check.generate_r1cs_constraints();
    const r1cs_constraint_system<FieldT> constraint_system = pb.get_constraint_system();
    return r1cs_gg_ppzksnark_generator<default_r1cs_gg_ppzksnark_pp>(constraint_system);
}

r1cs_gg_ppzksnark_proof<default_r1cs_gg_ppzksnark_pp> generate_proof(r1cs_gg_ppzksnark_proving_key<default_r1cs_gg_ppzksnark_pp> proving_key,int dim,vector<float> c,vector<float> s,vector<float> c_r,vector<int> r,FieldT ck,float cos,float euc){
    default_r1cs_gg_ppzksnark_pp::init_public_params();
    protoboard<FieldT> pb;
    valid_check<FieldT> check(pb, dim);
    check.generate_r1cs_constraints();
    check.generate_r1cs_witness(c,s,c_r,r,ck,cos,euc);
    return r1cs_gg_ppzksnark_prover<default_r1cs_gg_ppzksnark_pp>(proving_key, pb.primary_input(), pb.auxiliary_input());
}

bool verify_proof(r1cs_gg_ppzksnark_verification_key<default_r1cs_gg_ppzksnark_pp> verification_key,r1cs_gg_ppzksnark_proof<default_r1cs_gg_ppzksnark_pp> proof,vector<float> s,vector<float> c_r,FieldT ck,float cos,float euc){
    default_r1cs_gg_ppzksnark_pp::init_public_params();
    r1cs_primary_input<FieldT> input;
    int i_k=65536,i_2k=256;
    for (auto x : s) {
        input.push_back(FieldT(x*i_k));
    }
    for (auto x : c_r) {
        input.push_back(FieldT(x*i_k));
    }
    input.push_back(FieldT(cos*i_2k));
    input.push_back(FieldT(euc*i_k));
    input.push_back(ck);
    return r1cs_gg_ppzksnark_verifier_strong_IC<default_r1cs_gg_ppzksnark_pp>(verification_key, input, proof);
}

FieldT get_check_value(vector<int> r){
    int j=0;
    int dim=r.size();
    FieldT tmp[10],mul[9];
    for(int i=0;i<10;i++){
      tmp[i]=0;
      if(i<9){
        mul[i]=0;
      }
    }
    for(int i=0;i<dim;i++){
      switch (j)
      {
      case 0:
        tmp[j]=tmp[j]+FieldT(r[i]);
        j++;
        break;
      case 1:
        tmp[j]=tmp[j]+FieldT(r[i]);
        j++;
        break;
      case 2:
        tmp[j]=tmp[j]+FieldT(r[i]);
        j++;
        break;
      case 3:
        tmp[j]=tmp[j]+FieldT(r[i]);
        j++;
        break;
      case 4:
        tmp[j]=tmp[j]+FieldT(r[i]);
        j++;
        break;
      case 5:
        tmp[j]=tmp[j]+FieldT(r[i]);
        j++;
        break;
      case 6:
        tmp[j]=tmp[j]+FieldT(r[i]);
        j++;
        break;
      case 7:
        tmp[j]=tmp[j]+FieldT(r[i]);
        j++;
        break;
      case 8:
        tmp[j]=tmp[j]+FieldT(r[i]);
        j++;
        break;
      case 9:
        tmp[j]=tmp[j]+FieldT(r[i]);
        j=0;
        break;
      default:
        break;
      }
    }
    mul[0]=tmp[0]*tmp[1];
    for(int i=1;i<9;i++){
      mul[i]=mul[i-1]*tmp[i+1];
    }
    return mul[8];
}


PYBIND11_MODULE(ZKP, m) {
    m.doc() = "A zkp library from cpp"; // optional module docstring

    
    // m.def("get_dim", &get_dim, "A function which get data dim");
    m.def("generate_keypairs", &generate_keypairs, "A function which generate_keypairs");
    m.def("generate_proof", &generate_proof, "A function which generate_proof");
    m.def("verify_proof", &verify_proof, "A function which verify_proof");
    m.def("get_check_value", &get_check_value, "A function which get_check_value");
    // m.def("pk_serialize", &pk_serialize, "A function which serialize pk");
    // m.def("proof_serialize", &proof_serialize, "A function which serialize proof");
    // m.def("pk_deserialize", &pk_deserialize, "A function which deserialize pk");
    // m.def("proof_deserialize", &proof_deserialize, "A function which deserialize proof");
    py::class_<r1cs_gg_ppzksnark_proving_key<default_r1cs_gg_ppzksnark_pp>>(m, "r1cs_gg_ppzksnark_proving_key");
    py::class_<r1cs_gg_ppzksnark_verification_key<default_r1cs_gg_ppzksnark_pp>>(m, "r1cs_gg_ppzksnark_verification_key");
    py::class_<r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp>>(m, "r1cs_gg_ppzksnark_keypair").def_readwrite("pk", &r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp>::pk).def_readwrite("vk", &r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp>::vk);
    py::class_<r1cs_gg_ppzksnark_proof<default_r1cs_gg_ppzksnark_pp>>(m, "r1cs_gg_ppzksnark_proof");
    py::class_<FieldT>(m, "FieldT");

}
