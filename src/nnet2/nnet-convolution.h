// nnet2/nnet-convolution.h

// Copyright 2015  Xingyu Na

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


#ifndef KALDI_NNET2_NNET_CONVOLUTION_H_
#define KALDI_NNET2_NNET_CONVOLUTION_H_

#include "nnet2/nnet-component.h"

namespace kaldi {
namespace nnet2 {
  class ConvolutionComponent: public UpdatableComponent {
  public:
    ConvolutionalComponent(): UpdatableComponent(),
      patch_dim_(0), patch_step_(0), patch_stride_(0), is_gradient_(false)
    { }
    ~ConvolutionalComponent()
    { }
    // constructor using another component
    ConvolutionComponent(const ConvolutionComponent &component):  UpdatableComponent(component),
      filter_params_(component.filter_params_),
      bias_params_(component.bias_params_),
      is_gradient_(component.is_gradient_) {}
    // constructor using parameters
    ConvolutionComponent(const CuMatrixBase<BaseFloat> &filter_params,
    	                   const CuMatrixBase<BaseFloat> &bias_params,
    	                   BaseFloat learning_rate): UpdatableComponent(learning_rate),
      filter_params_(filter_params),
      bias_params_(bias_params) {
        KALDI_ASSERT(filter_params.NumRows() == bias_params.Dim() &&
    	   bias_params.Dim() != 0);
        is_gradient_ = false;
      }

    int32 InputDim() const;
    int32 OutputDim() const;
    void Init(BaseFloat learning_rate, int32 input_dim, int32 output_dim,
  	          int32 patch_dim, int32 patch_step, int32 patch_stride,
  	          BaseFloat param_stddev, BaseFloat bias_stddev);
    void Init(BaseFloat learning_rate, std::string matrix_filename);

    // resize the component, setting the parameters to zero, while
    // leaving any other configuration values the same
    void Resize(int32 input_dim, int32 output_dim);
    std::string Info() const;
    void InitFromString(std::string args);
    std::string Type() const { return "ConvolutionComponent"; }
    bool BackpropNeedsInput() const { return false; }
    bool BackpropNeedsOutput() const { return false; }
    using Component::Propagate; // to avoid name hiding
    void Propagate(const ChunkInfo &in_info,
  		             const ChunkInfo &out_info,
  		             const CuMatrixBase<BaseFloat> &in,
  		             CuMatrixBase<BaseFloat> *out) const;
    void Scale(BaseFloat scale);
    void Add(BaseFloat alpha, const ConvolutionComponent &other);
    void Backprop(const ChunkInfo &in_info,
  	              const ChunkInfo &out_info,
  	              const CuMatrixBase<BaseFloat> &in_value,
  	              const CuMatrixBase<BaseFloat> &out_value,
  	              const CuMatrixBase<BaseFloat> &out_deriv,
  	              Component *to_update_in,
  	              CuMatrix<BaseFloat> *in_deriv);
    void SetZero(bool treat_as_gradient);
    void Read(std::istream &is, bool binary);
    void Write(std::ostream &os, bool binary) const;
    BaseFloat DotProduct(const ConvolutionComponent &other) const;
    Component* Copy() const;
    void PerturbParams(BaseFloat stddev);
    virtual void SetParams(const VectorBase<BaseFloat> &bias,
                           const MatrixBase<BaseFloat> &filter);
    const CuVector<BaseFloat> &BiasParams() { return bias_params_; }
    const CuMatrix<BaseFloat> &LinearParams() { return filter_params_; }
    int32 GetParameterDim() const;
    void Update(const CuMatrixBase<BaseFloat> &in_value,
  	      const CuMatrixBase<BaseFloat> &out_deriv);

  protected:
    int32 patch_dim_;
    int32 patch_step_;
    int32 patch_stride_;

    const AffineComponent &operator = (const AffineComponent &other); // Disallow.
    CuMatrix<BaseFloat> filter_params_;
    CuMatrix<BaseFloat> bias_params_;

    /** Buffer of reshaped inputs:
     *  1row = vectorized rectangular feature patch,
     *  1col = dim over speech frames,
     *  std::vector-dim = patch-position
     */
    std::vector<CuMatrix<BaseFloat> > vectorized_feature_patches_;

    /** Buffer for backpropagation:
     *  derivatives in the domain of 'vectorized_feature_patches_',
     *  1row = vectorized rectangular feature patch,
     *  1col = dim over speech frames,
     *  std::vector-dim = patch-position
     */
    std::vector<CuMatrix<BaseFloat> > feature_patch_diffs_;
    bool is_gradient_;
  }
} // namespace nnet2
} // namespace kaldi

#endif
