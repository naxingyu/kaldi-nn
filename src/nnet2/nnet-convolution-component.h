
class ConvolutionComponent: public UpdatableComponent {
 public:
  ConvolutionalComponent(): UpdatableComponent(),
    patch_dim_(0), patch_step_(0), patch_stride_(0), is_gradient_(false)
  { }
  ~ConvolutionalComponent()
  { }

  // constructor using another component
  ConvolutionComponent(const ConvolutionComponent &component):
  UpdatableComponent(component),
    filter_params_(component.filter_params_),
    bias_params_(component.bias_params_),
    is_gradient_(component.is_gradient_) {}

  // constructor using parameters
  ConvolutionComponent(const CuMatrixBase<BaseFloat> &filter_params,
		       const CuMatrixBase<BaseFloat> &bias_params,
		       BaseFloat learning_rate):
  UpdatableComponent(learning_rate),
    filter_params_(filter_params),
    bias_params_(bias_params) {
      KALDI_ASSERT(filter_params.NumRows() == bias_params.Dim() &&
		   bias_params.Dim() != 0);
      is_gradient_ = false;
    }
  
  // aquire input dim
  int32 InputDim() const {
    int32 filter_dim = filter_params_.NumRows();
    int32 num_splice = filter_dim / patch_dim_;
    return patch_stride_ * num_splice;
  }

  // aquire output dim
  int32 OutputDim() const {
    int32 num_filters = filter_params_.NumCols();
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    return num_patches * num_filters;
  }

  // initialize the component using hyperparameters
  void Init(BaseFloat learning_rate,
	    int32 input_dim, int32 output_dim,
	    int32 patch_dim, int32 patch_step, int32 patch_stride,
	    BaseFloat param_stddev, BaseFloat bias_stddev) {
    UpdatableComponent::Init(learning_rate);
    patch_dim_ = patch_dim;
    patch_step_ = patch_step;
    patch_stride_ = patch_stride;
    int32 num_splice = input_dim / patch_stride;
    int32 filter_dim = num_splices * patch_dim;
    int32 num_patches = 1 + (patch_stride - patch_dim) / patch_step;
    int32 num_filters = output_dim / num_patches;
    KALDI_ASSERT(input_dim % patch_stride == 0);
    KALDI_ASSERT((patch_stride - patch_dim) % patch_step == 0);
    KALDI_ASSERT(output_dim % num_patches == 0);
    
    filter_params_.Resize(num_filters, filter_dim);
    bias_params_.Resize(num_filters);
    KALDI_ASSERT(param_stddev >= 0.0 && bias_stddev >= 0.0);
    filter_params_.SetRandn();
    filter_params_.Scale(param_stddev);
    bias_params_.SetRandn();
    bias_params_.Scale(bias_stddev);
  }

  // initialize the component using predefined matrix file
  void Init(BaseFloat learning_rate,
	    std::string matrix_filename) {
    UpdatableComponent::Init(learning_rate);
    CuMatrix<BaseFloat> mat;
    ReadKaldiObject(matrix_filename, &mat);
    KALDI_ASSERT(mat.NumCols() >= 2);
    int32 filter_dim = mat.NumCols() - 1, num_filters = mat.NumRows();
    filter_params_.Resize(num_filters, filter_dim);
    bias_params_.Resize(num_filters);
    filter_params_.CopyFromMat(mat.Range(0, num_filters, 0, filter_dim));
    bias_params_.CopyColFromMat(mat, filter_dim);
  }

  // resize the component, setting the parameters to zero, while
  // leaving any other configuration values the same
  void Resize(int32 input_dim, int32 output_dim) {
    KALDI_ASSERT(input_dim > 0 && output_dim > 0);
    int32 num_splice = input_dim / patch_stride_;
    int32 filter_dim = num_splices * patch_dim_;
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    int32 num_filters = output_dim / num_patches;
    KALDI_ASSERT(input_dim % patch_stride_ == 0);
    KALDI_ASSERT((patch_stride_ - patch_dim_) % patch_step_ == 0);
    KALDI_ASSERT(output_dim % num_patches == 0);
    filter_params_.Resize(num_filters, filter_dim);
    bias_params_.Resize(num_filters);
  }

  // display information about component
  std::string Info() const {
    std::stringstream stream;
    BaseFloat filter_params_size = static_cast<BaseFloat>(filter_params_.NumRows())
      * static_cast<BaseFloat>(filter_params_.NumCols());
    // BaseFloat linear_stddev, bias_stddev;
    int32 num_splice = InputDim() / patch_stride_;
    int32 filter_dim = num_splices * patch_dim_;
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    int32 num_filters = OutputDim() / num_patches;

    stream << Type() << ", input-dim=" << InputDim()
           << ", output-dim=" << OutputDim()
           << ", num-splice=" << num_splice
           << ", num-patches=" << num_patches
           << ", num-filters=" << num_filters
           << ", filter-dim=" << filter_dim
           << ", learning-rate=" << LearrningRate();
    return stream.str();
  }

  // initialize the component using configuration file
  void InitFromString(std::string args) {
    std::string orig_args(args);
    bool ok = true;
    BaseFloat learning_rate = learning_rate_;
    std::string matrix_filename;
    int32 input_dim = -1, output_dim = -1;
    int32 patch_dim = -1, patch_step = -1, patch_stride = -1;
    ParseFromString("learning_rate", &args, &learning_rate);
    if (ParseFromString("matrix", &args, &matrix_filename)) {
      // initialize from prefined parameter matrix
      Init(learning_rate, matrix_filename);
      if (ParseFromString("input-dim", &args, &input_dim))
	KALDI_ASSERT(input_dim == InputDim() &&
		     "input-dim mismatch vs. matrix.");
      if (ParseFromString("output-dim", &args, &output_dim))
	KALDI_ASSERT(output_dim == OutputDim() &&
		     "output-dim mismatch vs. matrix.");
    } else {
      // initialize from configuration
      ok = ok && ParseFromString("input-dim", &args, &input_dim);
      ok = ok && ParseFromString("output-dim", &args, &output_dim);
      ok = ok && ParseFromString("patch-dim", &args, &patch_dim);
      ok = ok && ParseFromString("patch-step", &args, &patch_step);
      ok = ok && ParseFromString("patch-stride", &args, &patch_stride);
      BaseFloat param_stddev = 1.0 / std::sqrt(input_dim), bias_stddev = 1.0;
      ParseFromString("param-stddev", &args, &param_stddev);
      ParseFromString("bias-stddev", &args, &bias_stddev);
      Init(learning_rate, input_dim, output_dim,
	   patch_dim, patch_step, patch_stride, param_stddev, bias_stddev);
    }
    if (!args.empty())
      KALDI_ERR << "Could not process these elements in initializer: " << args;
    if (!ok)
      KALDI_ERR << "Bad initializer " << orig_args;
  }

  std::string Type() const { return "ConvolutionComponent"; }
  bool BackpropNeedsInput() const { return false; }
  bool BackpropNeedsOutput() const { return false; }

  using Component::Propagate; // to avoid name hiding
  // propagation function
  void Propagate(const ChunkInfo &in_info,
		 const ChunkInfo &out_info,
		 const CuMatrixBase<BaseFloat> &in,
		 CuMatrixBase<BaseFloat> *out) const {
    in_info.CheckSize(in);
    out_info.CheckSize(*out);
    KALDI_ASSERT(in_info.NumChunks() == out_info.NumChunks());

    // dims
    int32 num_splice = InputDim() / patch_stride_;
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    int32 num_filters = filter_params_.NumRows();
    int32 num_frames = in_info.NumChunks();
    int32 filter_dim = filter_params_.NumCols();

    // prepare the buffers 
    if (vectorized_feature_patches_.size() == 0) {
      vectorized_feature_patches_.resize(num_patches);
      feature_patch_diffs_.resize(num_patches);
    }

    // vectorize the inputs
    for (int32 p = 0; p < num_patches; p++) {
      vectorized_feature_patches_[p].Resize(num_frames, filter_dim, kSetZero);
      // build-up a column selection mask:
      std::vector<int32> column_mask;
      for (int32 s = 0; s < num_splice; s++) {
        for (int32 d = 0; d < patch_dim_; d++) {
          column_mask.push_back(p * patch_step_ + s * patch_stride_ + d);
        }
      }
      KALDI_ASSERT(column_mask.size() == filter_dim);
      // select the columns
      vectorized_feature_patches_[p].CopyCols(in, column_mask);
    }

    // compute filter activations
    for (int32 p = 0; p < num_patches; p++) {
      CuSubMatrix<BaseFloat> tgt(out->ColRange(p * num_filters, num_filters));
      tgt.AddVecToRows(1.0, bias_params_, 0.0); // add bias
      // apply all filters
      tgt.AddMatMat(1.0, vectorized_feature_patches_[p], kNoTrans, filter_params, kTrans, 1.0);
    }
  }

  // scale the parameters
  void Scale(BaseFloat scale) {
    filter_params_.Scale(scale);
    bias_params_.Scale(scale);
  }

  // add another convolution component
  void Add(BaseFloat alpha, const ConvolutionComponent &other) {
    KALDI_ASSERT(other != NULL);
    filter_params_.AddMat(alpha, other->filter_params_);
    bias_params_.AddVec(alpha, other->bias_params_);
  }
  
  // back propagation function
  void Backprop(const ChunkInfo &in_info,
		const ChunkInfo &out_info,
		const CuMatrixBase<BaseFloat> &in_value,
		const CuMatrixBase<BaseFloat> &out_value,
		const CuMatrixBase<BaseFloat> &out_deriv,
		Component *to_update_in,
		CuMatrix<BaseFloat> *in_deriv) const {
    ConvolutionComponent *to_update = dynamic_cast<ConvolutionComponent*>(to_update_in);
    int32 num_splice = InputDim() / patch_stride_;
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    int32 num_filters = filter_params_.NumRows();
    int32 num_frames = in_info.NumChunks();
    int32 filter_dim = filter_params_.NumCols();

    // backpropagate to vector of matrices
    // (corresponding to position of a filter)
    for (int32 p = 0; p < num_patches; p++) {
      feature_patch_diffs_[p].Resize(num_frames, filter_dim, kSetZero); // reset
      CuSubMatrix<BaseFloat> out_deriv_patch(out_deriv.ColRange(p * num_filters, num_filters));
      feature_patch_diffs_[p].AddMatMat(1.0, out_diff_patch, kNoTrans, filter_params_, kNoTrans, 0.0);
    }

    // sum the derivatives into in_deriv, we will compensate #summands
    for (int32 p = 0; p < num_patches; p++) {
      for (int32 s = 0; s < num_splice; s++) {
        CuSubMatrix<BaseFloat> src(feature_patch_diffs_[p].ColRange(s * patch_dim_, patch_dim_));
        CuSubMatrix<BaseFloat> tgt(in_deriv->ColRange(p * patch_step_ + s * patch_stride_, patch_dim_));
        tgt.AddMat(1.0, src); // sum
      }
    }
  }

  void SetZero(bool treat_as_gradient) {
    if (treat_as_gradient) {
      SetLearningRate(1.0);
    }
    filter_params_.SetZero();
    bias_params_.SetZero();
    if (treat_as_gradient) {
      is_gradient_ = true;
    }
  }
  // update parameters
  void Update(const CuMatrixBase<BaseFloat> &in_value,
	      const CuMatrixBase<BaseFloat> &out_deriv) {
    // useful dims
    int32 num_patches = 1 + (patch_stride_ - patch_dim_) / patch_step_;
    int32 num_filters = filter_params_.NumRows();
    int32 filter_dim = filter_params_.NumCols();
    CuMatrix<BaseFloat> filters_grad;
    CuVector<BaseFloat> bias_grad;

    //
    // calculate the gradient
    //
    filters_grad.Resize(num_filters, filter_dim, kSetZero); // reset
    bias_grad.Resize(num_filters, kSetZero); // reset
    // use all the patches
    for (int32 p = 0; p < num_patches; p++) { // sum
      CuSubMatrix<BaseFloat> diff_patch(out_deriv.ColRange(p * num_filters, num_filters));
      filters_grad.AddMatMat(1.0, diff_patch, kTrans, vectorized_feature_patches_[p], kNoTrans, 1.0);
      bias_grad.AddRowSumMat(1.0, diff_patch, 1.0);
    }

    //
    // update
    // 
    filters_.AddMat(-learning_rate_, filters_grad_);
    bias_.AddVec(-learning_rate_, bias_grad_);
  }

  

      
 private:
  int32 patch_dim_;  
  int32 patch_step_;
  int32 patch_stride_;

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
};
