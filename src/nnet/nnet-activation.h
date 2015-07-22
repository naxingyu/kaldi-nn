// nnet/nnet-activation.h

// Copyright 2011-2013  Brno University of Technology (author: Karel Vesely)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#ifndef KALDI_NNET_NNET_ACTIVATION_H_
#define KALDI_NNET_NNET_ACTIVATION_H_

#include "nnet/nnet-component.h"
#include "cudamatrix/cu-math.h"
#include "cudamatrix/cu-rand.h"
#include "util/text-utils.h"

namespace kaldi {
namespace nnet1 {

class Softmax : public Component {
 public:
  Softmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Softmax()
  { }

  Component* Copy() const { return new Softmax(*this); }
  ComponentType GetType() const { return kSoftmax; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = e^x_j/sum_j(e^x_j)
    out->ApplySoftMaxPerRow(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // simply copy the error derivative
    // (ie. assume crossentropy error function, 
    // while in_diff contains (net_output-target) :
    // this is already derivative of the error with 
    // respect to activations of last layer neurons)
    in_diff->CopyFromMat(out_diff);
  }
};



class BlockSoftmax : public Component {
 public:
  BlockSoftmax(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~BlockSoftmax()
  { }

  Component* Copy() const { return new BlockSoftmax(*this); }
  ComponentType GetType() const { return kBlockSoftmax; }
  
  void InitData(std::istream &is) {
    // parse config
    std::string token,
      dims_str;
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<BlockDims>") is >> dims_str;
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (BlockDims)";
      is >> std::ws; // eat-up whitespace
    }
    // parse dims,
    if (!kaldi::SplitStringToIntegers(dims_str, ":", false, &block_dims))
      KALDI_ERR << "Invalid block-dims " << dims_str;
    // sanity check
    int32 sum = 0;
    for (int32 i=0; i<block_dims.size(); i++) {
      sum += block_dims[i];
    }
    KALDI_ASSERT(sum == OutputDim()); 
  }

  void ReadData(std::istream &is, bool binary) {
    ReadIntegerVector(is, binary, &block_dims);
    block_offset.resize(block_dims.size()+1, 0);
    for (int32 i = 0; i < block_dims.size(); i++) {
      block_offset[i+1] = block_offset[i] + block_dims[i];
    }
    // check
    KALDI_ASSERT(OutputDim() == block_offset[block_offset.size()-1]);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteIntegerVector(os, binary, block_dims);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // perform softmax per block:
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      CuSubMatrix<BaseFloat> in_bl = in.ColRange(block_offset[bl], block_dims[bl]);
      CuSubMatrix<BaseFloat> out_bl = out->ColRange(block_offset[bl], block_dims[bl]);
      // y = e^x_j/sum_j(e^x_j)
      out_bl.ApplySoftMaxPerRow(in_bl);
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // copy the error derivative:
    // (assuming we already got softmax-cross-entropy derivative in out_diff)
    in_diff->CopyFromMat(out_diff);
    
    // zero-out line-in-block, where sum different from zero,
    // process per block:
    for (int32 bl = 0; bl < block_dims.size(); bl++) {
      CuSubMatrix<BaseFloat> diff_bl = in_diff->ColRange(block_offset[bl], block_dims[bl]);
      CuVector<BaseFloat> row_sum(diff_bl.NumRows());
      row_sum.AddColSumMat(1.0, diff_bl, 0.0); // 0:keep, 1:zero-out
      // we'll scale rows by 0/1 masks
      CuVector<BaseFloat> row_diff_mask(row_sum);
      row_diff_mask.Scale(-1.0); // 0:keep, -1:zero-out
      row_diff_mask.Add(1.0); // 1:keep, 0:zero-out
      // here we should have only 0 and 1
      diff_bl.MulRowsVec(row_diff_mask);
    }
  }

  std::vector<int32> block_dims;
  std::vector<int32> block_offset;
};


class Normalize : public Component {
 public:
  Normalize(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  {kNormFloor = pow(2.0, -66); }
  ~Normalize()
  { }

  Component* Copy() const { return new Normalize(*this); }
  ComponentType GetType() const { return kNormalize; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // This component modifies the vector of activations by scaling it so that the
    // root-mean-square equals 1.0.
    out->CopyFromMat(in);
    CuVector<BaseFloat> in_norm(in.NumRows());
    in_norm.AddDiagMat2(1.0 / in.NumCols(), in, kNoTrans, 0.0);
    in_norm.ApplyFloor(kNormFloor);
    in_norm.ApplyPow(-0.5);
    out->MulRowsVec(in_norm);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // 
    CuVector<BaseFloat> in_norm(in.NumRows());
    in_norm.AddDiagMat2(1.0 / in.NumCols(), in, kNoTrans, 0.0);
    in_norm.ApplyFloor(kNormFloor);
    in_norm.ApplyPow(-0.5);
    in_diff->AddDiagVecMat(1.0, in_norm, out_diff, kNoTrans, 0.0);
    in_norm.ReplaceValue(1.0 / sqrt(kNormFloor), 0.0);
    in_norm.ApplyPow(3.0);
    CuVector<BaseFloat> dot_prod(in_diff->NumRows());
    dot_prod.AddDiagMatMat(1.0, out_diff, kNoTrans, in, kTrans, 0.0);
    dot_prod.MulElements(in_norm);
    in_diff->AddDiagVecMat(-1.0 / in.NumCols(), dot_prod, in, kNoTrans, 1.0);
  }

  BaseFloat kNormFloor;
};

/*
class Maxout : public Component {
 public:
  Maxout(int32 dim_in, int32 dim_out)
    : Component(dim_in, dim_out)
  { }
  ~Maxout()
  { }

  Component* Copy() const { return new Maxout(*this); }
  ComponentType GetType() const { return kMaxout; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = max(x)
    int32 group_size = dim_in / dim_out;
    for (int32 j = 0; j < dim_out_; j++) {
      CuSubMatrix<BaseFloat> pool(out->ColRange(j, 1));
      pool.Set(-1e20);
      for (int32 i = 0; i < group_size; i++)
	pool.Max(in.ColRange(j * group_size + i, 1));
    }
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    in_diff->DiffSigmoid(out, out_diff);
    int32 group_size = dim_in / dim_out;
    for (int32 j = 0; j < dim_out; j++) {
      CuSubMatrix<BaseFloat> out_j(out.ColRange(j, 1));
      for (int32 i = 0; i < group_size; i++) {
        CuSubMatrix<BaseFloat> in_i(in.ColRange(j * group_size + i, 1));
        CuSubMatrix<BaseFloat> in_d_i(in_diff.ColRange(j * group_size + i, 1));
        CuSubMatrix<BaseFloat> out_d_j(out_diff.ColRange(j, 1));
        CuMatrix<BaseFloat> mask;
        in_i.EqualElementMask(out_j, &mask);
        out_d_j.MulElements(mask);
        in_d_i.AddMat(1.0, out_d_j);
      }
    }
  }
};
*/

class Maxout : public Component {
 public:
  Maxout(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Maxout()
  { }

  Component* Copy() const { return new Maxout(*this); }
  ComponentType GetType() const { return kMaxout; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = max(x)
    out->GroupMax(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    in_diff->GroupMaxDeriv(in, out);
    in_diff->MulRowsGroupMat(out_diff);
  }
};

class Pnorm : public Component {
 public:
  Pnorm(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out), p_(2.0)
  { }
  ~Pnorm()
  { }

  Component* Copy() const { return new Pnorm(*this); }
  ComponentType GetType() const { return kPnorm; }

  void InitData(std::istream &is) {
    is >> std::ws; // eat-up whitespace
    // parse config
    std::string token;
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      if (token == "<P>") ReadBasicType(is, false, &p_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (P)";
    }
    KALDI_ASSERT(p_ >= 1.0);
  }

  void ReadData(std::istream &is, bool binary) {
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<P>");
      ReadBasicType(is, binary, &p_);
    }
    // check
    KALDI_ASSERT(p_ >= 1.0);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<P>");
    WriteBasicType(os, binary, p_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = (sum(|x^2|))^(1/p)
    out->GroupPnorm(in, p_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    in_diff->GroupPnormDeriv(in, out, p_);
    in_diff->MulRowsGroupMat(out_diff);
  }
 private:
  BaseFloat p_;
};


class Sigmoid : public Component {
 public:
  Sigmoid(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Sigmoid()
  { }

  Component* Copy() const { return new Sigmoid(*this); }
  ComponentType GetType() const { return kSigmoid; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = 1/(1+e^-x)
    out->Sigmoid(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    in_diff->DiffSigmoid(out, out_diff);
  }
};


class ReLU : public Component {
 public:
  ReLU(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~ReLU()
  { }

  Component* Copy() const { return new ReLU(*this); }
  ComponentType GetType() const { return kReLU; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = max(x, 0)
    out->CopyFromMat(in);
    out->ApplyFloor(0.0);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = (y > 0.0 ? 1.0 : 0.0)ex
    in_diff->CopyFromMat(out);
    in_diff->ApplyHeaviside();
    in_diff->MulElements(out_diff);
  }
};

class SoftHinge : public Component {
 public:
  SoftHinge(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~SoftHinge()
  { }

  Component* Copy() const { return new SoftHinge(*this); }
  ComponentType GetType() const { return kSoftHinge; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = log(1+e^x)
    out->SoftHinge(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = y(1-y)ex
    in_diff->Sigmoid(in);
    in_diff->MulElements(out_diff);
  }
};

class Tanh : public Component {
 public:
  Tanh(int32 dim_in, int32 dim_out) 
    : Component(dim_in, dim_out)
  { }
  ~Tanh()
  { }

  Component* Copy() const { return new Tanh(*this); }
  ComponentType GetType() const { return kTanh; }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    // y = (e^x - e^(-x)) / (e^x + e^(-x))
    out->Tanh(in);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    // ey = (1 - y^2)ex
    in_diff->DiffTanh(out, out_diff);
  }
};



class Dropout : public Component {
 public:
  Dropout(int32 dim_in, int32 dim_out):
      Component(dim_in, dim_out), dropout_retention_(0.5)
  { }
  ~Dropout()
  { }

  Component* Copy() const { return new Dropout(*this); }
  ComponentType GetType() const { return kDropout; }

  void InitData(std::istream &is) {
    is >> std::ws; // eat-up whitespace
    // parse config
    std::string token; 
    while (!is.eof()) {
      ReadToken(is, false, &token); 
      /**/ if (token == "<DropoutRetention>") ReadBasicType(is, false, &dropout_retention_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                     << " (DropoutRetention)";
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void ReadData(std::istream &is, bool binary) {
    if ('<' == Peek(is, binary)) {
      ExpectToken(is, binary, "<DropoutRetention>");
      ReadBasicType(is, binary, &dropout_retention_);
    }
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<DropoutRetention>");
    WriteBasicType(os, binary, dropout_retention_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    out->CopyFromMat(in);
    // switch off 50% of the inputs...
    dropout_mask_.Resize(out->NumRows(),out->NumCols());
    dropout_mask_.Set(dropout_retention_);
    rand_.BinarizeProbs(dropout_mask_,&dropout_mask_);
    out->MulElements(dropout_mask_);
    // rescale to keep same dynamic range as w/o dropout
    out->Scale(1.0/dropout_retention_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
    in_diff->CopyFromMat(out_diff);
    // use same mask on the error derivatives...
    in_diff->MulElements(dropout_mask_);
    // enlarge output to fit dynamic range w/o dropout
    in_diff->Scale(1.0/dropout_retention_);
  }
  
  BaseFloat GetDropoutRetention() {
    return dropout_retention_;
  }

  void SetDropoutRetention(BaseFloat dr) {
    dropout_retention_ = dr;
    KALDI_ASSERT(dropout_retention_ > 0.0 && dropout_retention_ <= 1.0);
  }

 private:
  CuRand<BaseFloat> rand_;
  CuMatrix<BaseFloat> dropout_mask_;
  BaseFloat dropout_retention_;
};



} // namespace nnet1
} // namespace kaldi

#endif

