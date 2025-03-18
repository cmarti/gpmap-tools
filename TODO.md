### High priority
- Implement contrast functionality in seqdeft under the laplace approximation (use Inverse Hessian as posterior covariance)
- Implement calculations with matrices that are better conditioned. Maybe the problems are in the diagonal terms from the likelihood.
- Review default regularization parameters in the grid in Minimim epistasis regression object with typically large spaces: the scale can also be very different, so it is unclear. Only reasonable calculation is to work from standardized data (or WT centered, and set priors in SD relative to the overall variance in the data: use data distribution to set it for instance)
- Review typos in documentation
- API section: need to have plotting functions and docstrings for DataSet methods
- Merge with master branch and release new stable version to pypi
- Write __str__ method for DataSet to highlight the main properties of the dataset
- Make clean and complete datasets
- Add data file with citations for the built-in datasets that is updated if new datasets are added with reference
- Fix and complete documentation about evolution on fitness landscapes
- Fix mixing time notation to relaxation and write the neutral relaxation time method for cases with mutational biases
- Try to implement nodes_vmin option for plotly interactive plotting to keep color scale constant even if we select top genotypes  
- Review bounds for plots in datashader allele grid: padding option may not have been activated. Maybe build some default function to use cross all datashader plots
- Review and update binaries with latests features
- Try to have a common function to evaluate predictions across subsets of data: evaluating and CV fitting
- Review SeqDEFT a range values. They don't flatten out at the ends as much as we would like.
- Add some more plotting details to the tutorials e.g. edge coloring and positioning of legends and colorbars in different places
- Make sure every function in the docs is in the API with docstrings as well
- Update plotly plotting functionalities
- Review Kernel alignment loss function in the paper, code and docs to match
- Add stationary function distribution to the histogram plot
- Try to reduce dependencies: networkx, datashader. Make some optional: tqdm, fastparquet, plotly
- Add documentation for transition path theory
- Think about separation of basic functionalities that are used in visualization and inference
- Explore why VC regression is so slow now and fix. Guess is the definition of the obs_idx matrix within the KernelOperator or the tolerance required by CG being to strict (most likely)


### Mid priority
- Change "a" parametrization to sigma or variance
- Add bin to rasterize nodes and edges
- Reorganize extended linear operator: think which methods should be functions to facilate class inheritance and which ones should work with matrices as well
- Reimplement datashader plotting functions using v0.12 native mpl interface
- There is a clear problem when calculating the rate matrix with mutational biases. Review and test if there is still error in calculation of rate matrix when having variable mutation rates. Re-test mutational biases
- Think whether to implement an embedding object -> This is almost the DataSet object now
- Streamline reduced alphabet
- Implement the 1SD rule to select hyperparameter in cv
- Implement DeltaP interpolation and regression
  
### Low priority
- Add some descriptions to the datasets and reference to the original paper
- Implement new likelihood functions for DeltaP based priors: poisson and binomial at least, maybe something similar to GLM for regression like approaches: have likelihood function classes with log-likelihood and gradient methods. Another interesting possiblity is the use of contrastive losses
- Implement MEM model: either directly or using VCregression backbone. Fit a through Kernel alignment. Have prior on kernel aligner for lambdas that forces it to have a DeltaP prior. Easier solution: optimize hyperparameter through CV under arbitrary log-likelihood functions. 
- Create new classes of Discrete spaces for the extra models to  generate visualizations from: smily face and tree-like
- Write a method to plot a phylogenetic tree on a landscape: test on the codon landscape
- Write a method to simulate evolution in the landscape: use the jump matrix and exponential waiting times. 
- Write a method to simulate sequence evolution on a phylogenetic tree using the previous function.
- Go back and review polynomial coefficients calculations precision with different methods. We lose precision with l=7 or 8 already with current implementation
- Figure out inconsistency between using the correct eigenvalues and getting wrong projections in the skewed kernel. So far it makes more sense to have the correct projection operators
- Optimize Laplacian matvec method for further efficiency if possible
- Figure out matrix free versions of Skewed Laplacian and Vj's projection operators

### Long term
- Implement evidence maximization
- Implement contrastive losses
- Implement laplace approximation for variances in SeqDEFT
- Re-implement HMC from Wei-Chia's code
- Add section to docs on the practical implementation of the operators as compared with traditional
  gaussian process approaches that use the dense Cholesky decomposition, which is intractable for 
  large datasets.
- Try MCMC on marginal likelihood
- Add star tree phylogenetic likelihood