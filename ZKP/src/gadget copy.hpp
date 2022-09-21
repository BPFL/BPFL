#include <libsnark/common/default_types/r1cs_gg_ppzksnark_pp.hpp>
#include <libsnark/zk_proof_systems/ppzksnark/r1cs_gg_ppzksnark/r1cs_gg_ppzksnark.hpp>
#include <libsnark/gadgetlib1/pb_variable.hpp>
#include <libsnark/gadgetlib1/gadgets/basic_gadgets.hpp>
#include<vector>
#include <sstream>
#include <string>

using namespace libsnark;
using namespace std;

template<typename FieldT>

class cos_gadget : public gadget<FieldT> {
private:
  vector<pb_variable<FieldT>> dot,A_l2,B_l2,dot_square,k_dot_square,A_B,A_B_bound,o_l2;
  vector<pb_variable_array<FieldT>> A_square;
  vector<pb_variable_array<FieldT>> B_square;
  vector<pb_variable_array<FieldT>> A_B_O;
  pb_variable_array<FieldT> one;
  vector<inner_product_gadget<FieldT>> inner_gadget;
  vector<comparison_gadget<FieldT>> com_gadget;
  pb_variable<FieldT> less, less_or_eq,bound_square,o_bound_square;
  

public:
  int m,n,k=10000;
  vector<pb_variable_array<FieldT>> A,B;
  pb_variable<FieldT> bound,zero,o_bound;
  

  cos_gadget(protoboard<FieldT> &pb,const vector<pb_variable_array<FieldT>> &A,const vector<pb_variable_array<FieldT>>&B,const pb_variable<FieldT>&Bd,const pb_variable<FieldT>&o_Bd,const int M,const int N):
  gadget<FieldT>(pb, "cos_gadget"),A(A),B(B),bound(Bd),o_bound(o_Bd),m(M),n(N)
  {
    // Allocate variables to protoboard
    // The strings (like "x") are only for debugging purposes
    dot.resize(m);
    A_l2.resize(m);
    B_l2.resize(m);
    o_l2.resize(m);

    dot_square.resize(m);
    k_dot_square.resize(m);
    A_square.resize(m);
    B_square.resize(m);
    A_B_O.resize(m);
    A_B.resize(m);
    A_B_bound.resize(m);
    // one.resize(m);
    one.allocate(this->pb,n,"one");
    for(int i=0;i<m;i++){
      dot[i].allocate(this->pb,"dot");
      A_l2[i].allocate(this->pb,"A_l2");
      B_l2[i].allocate(this->pb,"B_l2");
      o_l2[i].allocate(this->pb,"o_l2");
      A_square[i].allocate(this->pb,n,"A_square");
      B_square[i].allocate(this->pb,n,"B_square");
      A_B_O[i].allocate(this->pb,n,"A_B_O");
      
      dot_square[i].allocate(this->pb,"dot_square");
      k_dot_square[i].allocate(this->pb,"k_dot_square");
      A_B[i].allocate(this->pb,"A_B");
      A_B_bound[i].allocate(this->pb,"A_B_bound");
    }
    less.allocate(this->pb, "less"); // must have
    less_or_eq.allocate(this->pb, "less_or_eq");
    bound_square.allocate(this->pb,"bound_square");
    o_bound_square.allocate(this->pb,"o_bound_square");
    zero.allocate(this->pb,"zero");

  }

  void generate_r1cs_constraints()
  {
    this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(bound, bound, bound_square));
    this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(o_bound, o_bound, o_bound_square));
    for(int i=0;i<m;i++){
      inner_product_gadget<FieldT> compute_inner_product(this->pb, A[i], B[i], dot[i], "compute_inner_product");
      inner_gadget.push_back(compute_inner_product);
      comparison_gadget<FieldT> cmp_dot(this->pb,32, zero ,dot[i], less, less_or_eq, "cmp_dot");
      com_gadget.push_back(cmp_dot);
      for(int j=0;j<n;j++){
        this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(A[i][j], A[i][j], A_square[i][j]));
        this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(B[i][j], B[i][j], B_square[i][j]));
        this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(A[i][j]-B[i][j], A[i][j]-B[i][j], A_B_O[i][j]));
      }
      inner_product_gadget<FieldT> compute_sum_A(this->pb, A_square[i], one, A_l2[i], "compute_sum_A");
      inner_gadget.push_back(compute_sum_A);
      inner_product_gadget<FieldT> compute_sum_B(this->pb, B_square[i],one, B_l2[i], "compute_sum_B");
      inner_gadget.push_back(compute_sum_B);
      this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(dot[i], dot[i], dot_square[i]));
      this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(10000, dot_square[i], k_dot_square[i]));
      this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(A_l2[i], B_l2[i], A_B[i]));
      this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(A_B[i], bound_square, A_B_bound[i]));
      comparison_gadget<FieldT> cmp(this->pb,100, A_B_bound[i] ,k_dot_square[i], less, less_or_eq, "cmp_cos");
      com_gadget.push_back(cmp);
      inner_product_gadget<FieldT> compute_o_l2(this->pb, A_B_O[i], one, o_l2[i], "compute_o_l2");
      inner_gadget.push_back(compute_o_l2);
      comparison_gadget<FieldT> cmp_o(this->pb,100, o_l2[i] ,o_bound_square, less, less_or_eq, "cmp_o");
      com_gadget.push_back(cmp_o);
    }
    this->pb.add_r1cs_constraint(r1cs_constraint<FieldT>(less_or_eq, 1, FieldT::one()));
    for(int i=0;i<inner_gadget.size();i++){
      inner_gadget[i].generate_r1cs_constraints();
    }
    for(int i=0;i<com_gadget.size();i++){
      com_gadget[i].generate_r1cs_constraints();
    }
    
  }

  //   void get(){
  //   for(int i=0;i<m;i++){
  //     cout<<"A&B"<<endl;
  //     for(int j=0;j<n;j++){
  //       if(j==10){
  //         break;
  //       }
  //       cout<<this->pb.val(A[i][j])<<endl;
  //       cout<<this->pb.val(B[i][j])<<endl;
  //     }
  //     cout<<"dot_square_K"<<endl;
  //     cout<<this->pb.val(dot[i])<<endl;
  //     cout<<this->pb.val(dot_square[i])<<endl;
  //     cout<<this->pb.val(k_dot_square[i])<<endl;
      

  //     cout<<"l2"<<endl;
  //     cout<<this->pb.val(A_l2[i])<<endl;
  //     cout<<this->pb.val(B_l2[i])<<endl;

  //     cout<<"B"<<endl;
  //     cout<<this->pb.val(bound)<<endl;
  //     cout<<this->pb.val(bound_square)<<endl;

  //     cout<<"ABbound"<<endl;
  //     cout<<this->pb.val(A_B_bound[i])<<endl;
  //   }

  // }

  void generate_r1cs_witness()
  {
    this->pb.val(bound_square)=this->pb.val(bound)*this->pb.val(bound);
    this->pb.val(o_bound_square)=this->pb.val(o_bound)*this->pb.val(o_bound);
    // cout<<"o_bound_square------------"<<this->pb.val(o_bound_square)<<endl;
    this->pb.val(zero)=0;
    one.fill_with_field_elements(this->pb,vector<FieldT>(n,1));
    for(int i=0;i<m;i++){
      this->pb.val(dot[i]) =0;
      vector<FieldT> A_val=A[i].get_vals(this->pb);
      vector<FieldT> B_val=B[i].get_vals(this->pb);
      vector<FieldT> A_B_O_val;
      for(int j=0;j<n;j++){
        // cout<<"A"<<A_val[j]<<"B"<<B_val[j]<<endl;
        A_B_O_val.push_back((A_val[j]-B_val[j])*(A_val[j]-B_val[j]));
        // cout<<"ABO:::"<<A_B_O_val[j]<<endl;
      }
      for(int j=0;j<n;j++){
        this->pb.val(dot[i])+=this->pb.val(A[i][j])*this->pb.val(B[i][j]);
        A_val[j]*=A_val[j];
        B_val[j]*=B_val[j];
      }
      
      A_square[i].fill_with_field_elements(this->pb,A_val);
      B_square[i].fill_with_field_elements(this->pb,B_val);
      A_B_O[i].fill_with_field_elements(this->pb,A_B_O_val);

      for(int j=0;j<n;j++){
        this->pb.val(A_l2[i])+=A_val[j];
        this->pb.val(B_l2[i])+=B_val[j];
        this->pb.val(o_l2[i])+=A_B_O_val[j];
        
      }
      // cout<<"ol2::::::::::::"<<i<<"::::::"<<this->pb.val(o_l2[i])<<endl;
      for(int i=0;i<inner_gadget.size();i++){
        inner_gadget[i].generate_r1cs_witness();
      }
      this->pb.val(dot_square[i])=this->pb.val(dot[i])*this->pb.val(dot[i]);
      this->pb.val(k_dot_square[i])=this->pb.val(dot_square[i])*10000;
      this->pb.val(A_B[i])=this->pb.val(A_l2[i])*this->pb.val(B_l2[i]);
      this->pb.val(A_B_bound[i])=this->pb.val(A_B[i])*this->pb.val(bound_square);
      for(int i=0;i<com_gadget.size();i++){
        com_gadget[i].generate_r1cs_witness();
      }
    }
    //get();
  }


};

template<typename FieldT>
class pack_cos_gadget{
  private:
  vector<cos_gadget<FieldT>> cos_gadget_unit;
  vector<vector<vector<pb_variable<FieldT>>>> sub_A_A_M,multi_sub;
  public:
  vector<vector<int>> Dim;
  vector<vector<pb_variable_array<FieldT>>>A, B,A_M;
  pb_variable<FieldT> bound,o_bound;
  pb_variable<FieldT> q;
  pb_variable<FieldT> E;
  pb_variable<FieldT> M;
  pb_variable<FieldT> q_M;

  pack_cos_gadget(protoboard<FieldT> &pb,const vector<vector<int>> &dim):Dim(dim)
  {
    int total=0;
    A.resize(Dim.size());
    B.resize(Dim.size());
    sub_A_A_M.resize(Dim.size());
    multi_sub.resize(Dim.size());
    A_M.resize(Dim.size());
    bound.allocate(pb,"bound"); 
    o_bound.allocate(pb,"o_bound"); 
    q.allocate(pb,"q"); 
    E.allocate(pb,"E");  
    for(int i=0;i<Dim.size();i++){
      A_M[i].resize(Dim[i][0]);
      sub_A_A_M[i].resize(Dim[i][0]);
      multi_sub[i].resize(Dim[i][0]);
      for(int j=0;j<Dim[i][0];j++){
        A_M[i][j].allocate(pb,Dim[i][1],"A_M");

        sub_A_A_M[i][j].resize(Dim[i][1]);
        multi_sub[i][j].resize(Dim[i][1]);
      }
    }
    for(int i=0;i<Dim.size();i++){
      B[i].resize(Dim[i][0]);
      for(int j=0;j<Dim[i][0];j++){
        B[i][j].allocate(pb,Dim[i][1],"B");
      }
      total+=dim[i][0]*Dim[i][1]*2;
    }
    for(int i=0;i<Dim.size();i++){
      for(int j=0;j<Dim[i][0];j++){
        for(int k=0;k<Dim[i][1];k++){
          sub_A_A_M[i][j][k].allocate(pb,"sub_A_A_M"); 
          multi_sub[i][j][k].allocate(pb,"multi_sub"); 
          // pb.val(sub_A_A_M[i][j][k])=pb.val(A[i][j][k])+pb.val(M)-pb.val(A_M[i][j][k]);
          // pb.val(multi_sub)=(pb.val(A[i][j][k])+pb.val(M)-pb.val(A_M[i][j][k]))*(pb.val(A[i][j][k])+pb.val(M)-pb.val(A_M[i][j][k]));
        }
      }
    }
    pb.set_input_sizes(total+4);
    M.allocate(pb,"M");  
    q_M.allocate(pb,"q_M");
    for(int i=0;i<Dim.size();i++){
      A[i].resize(Dim[i][0]);
      for(int j=0;j<Dim[i][0];j++){
        A[i][j].allocate(pb,dim[i][1],"A");
      }
    }
    
    for(int i=0;i<Dim.size();i++){
        cos_gadget<FieldT> cos(pb,A[i],B[i],bound,o_bound,dim[i][0],dim[i][1]);
        cos_gadget_unit.push_back(cos);
    }
    
   
  }

  void generate_r1cs_constraints(protoboard<FieldT> &pb){
    pb.add_r1cs_constraint(r1cs_constraint<FieldT>(q_M,1, E));
    
    for(int i=0;i<Dim.size();i++){
        for(int j=0;j<Dim[i][0];j++){
          for(int k=0;k<Dim[i][1];k++){
            // pb.add_r1cs_constraint(r1cs_constraint<FieldT>(A[i][j][k]+M-A_M[i][j][k],1-(A[i][j][k]+M-A_M[i][j][k]), 0));
            pb.add_r1cs_constraint(r1cs_constraint<FieldT>(A[i][j][k]+M-A_M[i][j][k],1, sub_A_A_M[i][j][k]));
            pb.add_r1cs_constraint(r1cs_constraint<FieldT>(sub_A_A_M[i][j][k],sub_A_A_M[i][j][k], multi_sub[i][j][k]));
            pb.add_r1cs_constraint(r1cs_constraint<FieldT>(multi_sub[i][j][k], 1-(multi_sub[i][j][k]), 0));
          }
        }
    }

    for(int i=0;i<Dim.size();i++){
        cos_gadget_unit[i].generate_r1cs_constraints();
    }
  }

  void generate_r1cs_witness(protoboard<FieldT> &pb,const vector<vector<vector<float>>>&x,const vector<vector<vector<float>>>&y,const vector<vector<vector<float>>>&x_m,const int m,const int qq,const unsigned long long e,const float &Bd,const float &o_Bd){
    pb.val(bound)=Bd*100;
    pb.val(o_bound)=o_Bd*10000;
    pb.val(M)=m*10000;
    pb.val(q_M)=1;
    pb.val(q)=qq;
    pb.val(E)=e;
    // cout<<"o_bound------------"<<pb.val(o_bound)<<endl;
    for(int i=0;i<m;i++){
      pb.val(q_M)*=pb.val(q);
    }
    
    for(int i=0;i<Dim.size();i++){
      for(int j=0;j<Dim[i][0];j++){
        vector<FieldT> data_x,data_y,data_x_m;
        for(int k=0;k<Dim[i][1];k++){
          data_x.push_back(x[i][j][k]*10000);
          data_y.push_back(y[i][j][k]*10000);
          data_x_m.push_back(x_m[i][j][k]*10000);
        }
        A[i][j].fill_with_field_elements(pb,data_x);
        B[i][j].fill_with_field_elements(pb,data_y);
        A_M[i][j].fill_with_field_elements(pb,data_x_m);
      }
    }
    for(int i=0;i<Dim.size();i++){
      for(int j=0;j<Dim[i][0];j++){
        for(int k=0;k<Dim[i][1];k++){
          // sub_A_A_M.allocate(pb,"sub_A_A_M"); 
          // multi_sub.allocate(pb,"multi_sub"); 
          FieldT sub=pb.val(A[i][j][k])+pb.val(M)-pb.val(A_M[i][j][k]);
          pb.val(sub_A_A_M[i][j][k])=sub;
          pb.val(multi_sub[i][j][k])=sub*sub;
        }
      }
    }

    // for(int i=0;i<Dim.size();i++){
    //     for(int j=0;j<Dim[i][0];j++){
    //       for(int k=0;k<Dim[i][1];k++){
    //         if(!(pb.val(multi_sub[i][j][k])==1||pb.val(multi_sub[i][j][k])==0))
    //           cout<<pb.val(sub_A_A_M[i][j][k])<<"---------"<<pb.val(multi_sub[i][j][k])<<"----------------"<<pb.val(multi_sub[i][j][k])<<endl;
    //           // cout<<pb.val(A[i][j][k])+pb.val(M)-pb.val(A_M[i][j][k])<<"---------"<<endl;

    //       }
    //     }
    // }
    
    for(int i=0;i<cos_gadget_unit.size();i++){
      cos_gadget_unit[i].generate_r1cs_witness();
    }
  }
};