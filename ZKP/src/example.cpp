#include <pybind11/pybind11.h>
#include <vector>
#include <iostream>
#include <pybind11/stl.h>
#include <libsnark/common/default_types/r1cs_gg_ppzksnark_pp.hpp>
#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>
#include <libsnark/gadgetlib1/pb_variable.hpp>
#include <libsnark/gadgetlib1/gadgets/basic_gadgets.hpp>

namespace py = pybind11;
using namespace libsnark;
using namespace std;

typedef libff::Fr<default_r1cs_gg_ppzksnark_pp> FieldT;

r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp> ge_key() {
    typedef libff::Fr<default_r1cs_gg_ppzksnark_pp> FieldT;

    // Initialize the curve parameters
    default_r1cs_gg_ppzksnark_pp::init_public_params();
  
    // Create protoboard
    protoboard<FieldT> pb;

    pb_variable<FieldT> x, max;
    pb_variable<FieldT> less, less_or_eq;

    x.allocate(pb, "x");
    max.allocate(pb, "max");
    less.allocate(pb, "less"); // must have
    less_or_eq.allocate(pb, "less_or_eq");
    pb.set_input_sizes(1);
    
    // pb.val(max)= 60;

    comparison_gadget<FieldT> cmp(pb, 10, x, max, less, less_or_eq, "cmp");
    cmp.generate_r1cs_constraints();
    pb.add_r1cs_constraint(r1cs_constraint<FieldT>(less, 1, FieldT::one()));

    const r1cs_constraint_system<FieldT> constraint_system = pb.get_constraint_system();

    // generate keypair
    const r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp> keypair = r1cs_gg_ppzksnark_generator<default_r1cs_gg_ppzksnark_pp>(constraint_system);
    return keypair;
}

r1cs_gg_ppzksnark_proof<default_r1cs_gg_ppzksnark_pp> ge_proof(r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp> keypair){
    typedef libff::Fr<default_r1cs_gg_ppzksnark_pp> FieldT;

    // Initialize the curve parameters
    default_r1cs_gg_ppzksnark_pp::init_public_params();
  
    // Create protoboard
    protoboard<FieldT> pb;

    pb_variable<FieldT> x, max;
    pb_variable<FieldT> less, less_or_eq;

    x.allocate(pb, "x");
    max.allocate(pb, "max");
    less.allocate(pb, "less"); // must have
    less_or_eq.allocate(pb, "less_or_eq");
    pb.set_input_sizes(1);
    
    pb.val(max)= 60;

    comparison_gadget<FieldT> cmp(pb, 10, x, max, less, less_or_eq, "cmp");
    cmp.generate_r1cs_constraints();
    pb.add_r1cs_constraint(r1cs_constraint<FieldT>(less, 1, FieldT::one()));
    // Add witness values
    pb.val(x) = 18; // secret
    cmp.generate_r1cs_witness();
    // cout << "Primary (public) input: " << pb.primary_input() << endl;
    // cout << "Auxiliary (private) input: " << pb.auxiliary_input() << endl;

    // generate proof
    const r1cs_gg_ppzksnark_proof<default_r1cs_gg_ppzksnark_pp> proof = r1cs_gg_ppzksnark_prover<default_r1cs_gg_ppzksnark_pp>(keypair.pk, pb.primary_input(), pb.auxiliary_input());
    return proof;
}

int ge_ver(r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp> keypair,r1cs_gg_ppzksnark_proof<default_r1cs_gg_ppzksnark_pp> proof){
    r1cs_primary_input<FieldT> input;
    input.push_back(FieldT(18));
    bool verified = r1cs_gg_ppzksnark_verifier_strong_IC<default_r1cs_gg_ppzksnark_pp>(keypair.vk, input, proof);
    cout << "Verification status: " << verified << endl;
    return int(verified);
}


PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("ge_key", &ge_key, "A function which adds two numbers");
    m.def("ge_proof", &ge_proof, "A function which adds two numbers");
    m.def("ge_ver", &ge_ver, "A function which adds two numbers");
    py::class_<r1cs_gg_ppzksnark_proving_key<default_r1cs_gg_ppzksnark_pp>>(m, "r1cs_gg_ppzksnark_proving_key");
    py::class_<r1cs_gg_ppzksnark_verification_key<default_r1cs_gg_ppzksnark_pp>>(m, "r1cs_gg_ppzksnark_verification_key");
    py::class_<r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp>>(m, "r1cs_gg_ppzksnark_keypair").def_readwrite("pk", &r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp>::pk).def_readwrite("vk", &r1cs_gg_ppzksnark_keypair<default_r1cs_gg_ppzksnark_pp>::vk);
    py::class_<r1cs_gg_ppzksnark_proof<default_r1cs_gg_ppzksnark_pp>>(m, "r1cs_gg_ppzksnark_proof");
}