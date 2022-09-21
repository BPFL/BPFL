#include <libsnark/common/default_types/r1cs_gg_ppzksnark_pp.hpp>
#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>
#include <libsnark/gadgetlib1/pb_variable.hpp>
#include <libsnark/gadgetlib1/gadgets/basic_gadgets.hpp>
#include<vector>
#include <sstream>
#include <string>

using namespace libsnark;
using namespace std;
typedef libff::Fr<default_r1cs_gg_ppzksnark_pp> FieldT;
template<typename FieldT>

class valid_check: public gadget<FieldT> {
private:
  pb_variable_array<FieldT> one,sub_mask,sub_S_C;
  pb_variable<FieldT> zero,k,inner,l_s,l_c,inner_sq,k_inner_sq,cos_sq,l_s_c,cos_l,euc_sq,sub_l,less, less_or_eq;
  vector<inner_product_gadget<FieldT>> inner_gadget;
  vector<comparison_gadget<FieldT>> com_gadget;
  pb_variable<FieldT> tmp[10],mul[9],cmp_cos_value,cmp_euc_value;
public:
  int dim;//2^12=4096  2^24=16777216
  // int i_k=10000,i_2k=100;
  int i_k=65536,i_2k=256;
  pb_variable_array<FieldT> S,C,M_C,M;
  pb_variable<FieldT> cos,euc,check_value;
  

  valid_check(protoboard<FieldT> &pb,int dim):
  gadget<FieldT>(pb, "valid_check"),dim(dim)
  {
    // Allocate variables to protoboard
    // The strings (like "x") are only for debugging purposes
    S.allocate(this->pb,dim,"S");
    M_C.allocate(this->pb,dim,"M_C");
    cos.allocate(this->pb,"cos");
    euc.allocate(this->pb,"euc");
    check_value.allocate(this->pb,"check_value");
    pb.set_input_sizes(dim*2+3);
    for(int i=0;i<10;i++){
      tmp[i].allocate(this->pb,"tmp");
      if(i<9){
        mul[i].allocate(this->pb,"mul");
      }
    }
    C.allocate(this->pb,dim,"C");
    M.allocate(this->pb,dim,"M");
    one.allocate(this->pb,dim,"one");
    sub_mask.allocate(this->pb,dim,"sub_mask");
    sub_S_C.allocate(this->pb,dim,"sub_s_c");
    zero.allocate(this->pb,"zero");
    k.allocate(this->pb,"k");
    inner.allocate(this->pb,"inner");
    l_s.allocate(this->pb,"l_s");
    l_c.allocate(this->pb,"l_c");
    inner_sq.allocate(this->pb,"inner_sq");
    k_inner_sq.allocate(this->pb,"k_inner_sq");
    cos_sq.allocate(this->pb,"cos_sq");
    l_s_c.allocate(this->pb,"l_s_c");
    cos_l.allocate(this->pb,"cos_l");
    euc_sq.allocate(this->pb,"euc_sq");
    sub_l.allocate(this->pb,"sub_l");
    less.allocate(this->pb, "less"); // must have
    less_or_eq.allocate(this->pb, "less_or_eq");
    cmp_cos_value.allocate(this->pb, "cmp_cos_value");
    cmp_euc_value.allocate(this->pb, "cmp_euc_value");

  }

  void generate_r1cs_constraints()
  {
    //cos
    inner_product_gadget<FieldT> s_c_inner_product(this->pb, S, C, inner, "s_c_inner_product");
    inner_gadget.push_back(s_c_inner_product);
    comparison_gadget<FieldT> cmp_inner(this->pb,64, zero ,inner, less, less_or_eq, "cmp_inner");
    com_gadget.push_back(cmp_inner);
    this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(inner, inner, inner_sq));
    this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(inner_sq, k, k_inner_sq));
    inner_product_gadget<FieldT> s_inner_product(this->pb, S, S, l_s, "s_l");
    inner_gadget.push_back(s_inner_product);
    inner_product_gadget<FieldT> c_inner_product(this->pb, C, C, l_c, "c_l");
    inner_gadget.push_back(c_inner_product);
    this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(cos, cos, cos_sq));
    this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(l_s,l_c, l_s_c));
    this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(l_s_c,cos_sq, cos_l));
    comparison_gadget<FieldT> cmp_cos(this->pb,200, zero,cmp_cos_value, less, less_or_eq, "cmp_cos");
    com_gadget.push_back(cmp_cos);
    
    // // //euc
    inner_product_gadget<FieldT> euc_inner_product(this->pb, sub_S_C, sub_S_C, sub_l, "sub_euc_l");
    inner_gadget.push_back(euc_inner_product);
    this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(euc, euc, euc_sq));
    comparison_gadget<FieldT> cmp_euc(this->pb,64, zero,cmp_euc_value, less, less_or_eq, "cmp_euc");
    com_gadget.push_back(cmp_euc);
    // // check mask
    for(size_t i=0;i<dim;++i){
      // this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(M_C[i]-C[i]-M[i], 1, 0));
      // this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>((C[i]+M[i])-M_C[i], 1, sub_mask[i]));
      this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(sub_mask[i], 1-sub_mask[i], 0));
    }
    // // verify mask
    for(int i=1;i<9;i++){
      this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(mul[i-1], tmp[i+1], mul[i]));
    }
    this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(mul[8], 1, check_value));
    for(int i=0;i<inner_gadget.size();i++){
      inner_gadget[i].generate_r1cs_constraints();
    }
    for(int i=0;i<com_gadget.size();i++){
      com_gadget[i].generate_r1cs_constraints();
    }
    this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(less_or_eq, 1, FieldT::one()));
  }

  void generate_r1cs_witness(vector<float> x,vector<float> y,vector<float> x_m,vector<int> r,FieldT ck,float cos_value,float euc_value)
  {
    // this->pb.val(less_or_eq)=1;
    this->pb.val(k)=i_k;
    for(int i=0;i<dim;i++){
      this->pb.val(C[i])=x[i]*i_k;
      this->pb.val(S[i])=y[i]*i_k;
      this->pb.val(M_C[i])=x_m[i]*i_k;
      this->pb.val(M[i])=r[i]*i_k;
    }
    this->pb.val(cos)=cos_value*i_2k;
    this->pb.val(euc)=euc_value*i_k;
    this->pb.val(check_value)=ck;

    int j=0;
    for(int i=0;i<dim;i++){
      switch (j)
      {
      case 0:
        this->pb.val(tmp[j])=this->pb.val(tmp[j])+r[i];
        j++;
        break;
      case 1:
        this->pb.val(tmp[j])=this->pb.val(tmp[j])+r[i];
        j++;
        break;
      case 2:
        this->pb.val(tmp[j])=this->pb.val(tmp[j])+r[i];
        j++;
        break;
      case 3:
        this->pb.val(tmp[j])=this->pb.val(tmp[j])+r[i];
        j++;
        break;
      case 4:
        this->pb.val(tmp[j])=this->pb.val(tmp[j])+r[i];
        j++;
        break;
      case 5:
        this->pb.val(tmp[j])=this->pb.val(tmp[j])+r[i];
        j++;
        break;
      case 6:
        this->pb.val(tmp[j])=this->pb.val(tmp[j])+r[i];
        j++;
        break;
      case 7:
        this->pb.val(tmp[j])=this->pb.val(tmp[j])+r[i];
        j++;
        break;
      case 8:
        this->pb.val(tmp[j])=this->pb.val(tmp[j])+r[i];
        j++;
        break;
      case 9:
        this->pb.val(tmp[j])=this->pb.val(tmp[j])+r[i];
        j=0;
        break;
      default:
        break;
      }
    }
    this->pb.val(mul[0])=this->pb.val(tmp[0])*this->pb.val(tmp[1]);
    for(int i=1;i<9;i++){
      this->pb.val(mul[i])=this->pb.val(mul[i-1])*this->pb.val(tmp[i+1]);
    }
    this->pb.val(zero)=0;
    this->pb.val(inner)=0;
    this->pb.val(l_s)=0;
    this->pb.val(l_c)=0;
    for(int i=0;i<dim;i++){
      this->pb.val(one[i])=1;
      this->pb.val(sub_S_C[i])=this->pb.val(S[i])-this->pb.val(C[i]);
      this->pb.val(inner)=this->pb.val(inner)+(this->pb.val(S[i])*this->pb.val(C[i]));
      this->pb.val(l_s)=this->pb.val(l_s)+(this->pb.val(S[i])*this->pb.val(S[i]));
      this->pb.val(l_c)=this->pb.val(l_c)+(this->pb.val(C[i])*this->pb.val(C[i]));
      this->pb.val(sub_l)=this->pb.val(sub_l)+(this->pb.val(sub_S_C[i])*this->pb.val(sub_S_C[i]));
      this->pb.val(sub_mask[i])=(this->pb.val(C[i])+this->pb.val(M[i]))-this->pb.val(M_C[i]);
      this->pb.val(sub_mask[i])=this->pb.val(sub_mask[i])*this->pb.val(sub_mask[i]);
    }
    this->pb.val(inner_sq)=this->pb.val(inner)*this->pb.val(inner);
    this->pb.val(k_inner_sq)=this->pb.val(k)*this->pb.val(inner_sq);
    this->pb.val(cos_sq)=this->pb.val(cos)*this->pb.val(cos);
    this->pb.val(l_s_c)=this->pb.val(l_s)*this->pb.val(l_c);
    this->pb.val(cos_l)=this->pb.val(cos_sq)*this->pb.val(l_s_c);
    this->pb.val(euc_sq)=this->pb.val(euc)*this->pb.val(euc);
    this->pb.val(cmp_cos_value)=this->pb.val(k_inner_sq)-this->pb.val(cos_l);
    this->pb.val(cmp_euc_value)=this->pb.val(euc_sq)-this->pb.val(sub_l);
    for(int i=0;i<inner_gadget.size();i++){
      inner_gadget[i].generate_r1cs_witness();
    }
    for(int i=0;i<com_gadget.size();i++){
      com_gadget[i].generate_r1cs_witness();
    }
    //  get();
  }
  // void get(){
  //   for(int i=0;i<dim;i++){
  //     cout<<this->pb.val(C[i])<<" "<<this->pb.val(M_C[i])<<" "<<this->pb.val(sub_mask[i])<<endl;

  //   }
  //   for(int i=0;i<10;i++){
  //     cout<<this->pb.val(tmp[i])<<endl;
  //   }
  //   for(int i=0;i<9;i++){
  //     cout<<this->pb.val(mul[i])<<endl;
  //   }
  //   cout<<"inner       "<<this->pb.val(inner)<<endl;
  //   cout<<"l_s         "<<this->pb.val(l_s)<<endl;
  //   cout<<"l_c         "<<this->pb.val(l_c)<<endl;
  //   cout<<"inner_sq    "<<this->pb.val(inner_sq)<<endl;
  //   cout<<"k_inner_sq  "<<this->pb.val(k_inner_sq)<<endl;
  //   cout<<"cos_sq      "<<this->pb.val(cos_sq)<<endl;
  //   cout<<"l_s_c       "<<this->pb.val(l_s_c)<<endl;
  //   cout<<"cos_l       "<<this->pb.val(cos_l)<<endl;
  //   cout<<"euc_sq      "<<this->pb.val(euc_sq)<<endl;
  //   cout<<"sub_l       "<<this->pb.val(sub_l)<<endl;
  //   cout<<"cos         "<<this->pb.val(cos)<<endl;
  //   cout<<"euc         "<<this->pb.val(euc)<<endl;
  //   cout<<"check_value "<<this->pb.val(check_value)<<endl;
  //   cout<<"mul8        "<<this->pb.val(mul[8])<<endl;
  //   cout<<"cpm_cos     "<<this->pb.val(cmp_cos_value)<<endl;
  //   cout<<"cmp_euc     "<<this->pb.val(cmp_euc_value)<<endl;
    
  //   for(int i=0;i<dim;i++){
  //     if(!(this->pb.val(sub_mask[i])==0||this->pb.val(sub_mask[i])==1))
  //     cout<<this->pb.val(sub_mask[i])<<endl;

  //   }
  // }
};