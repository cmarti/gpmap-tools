### High priority
- Try to implement nodes_vmin option for plotly interactive plotting to keep color scale constant even if we select top genotypes  
- Review bounds for plots in datashader allele grid: padding option may not have been activated. Maybe build some default function to use cross all datashader plots
- Clean up some options for plotting that we are not using that much and make everything confusing.
- Have default figure configurations in some global variables or configuration that can be easily changed across all plots
- Add function to plot path using nodes_df and a list of sequences
- Merge with master branch and release new stable version to pypi
- Clean warnings and prints from running tests
- Remove tests dependencies on data outside the datasets. Use simulations
- Fix plot tests
- Change "a" parametrization to sigma or variance
- Review and update binaries with latests features
- Try to have a common function to evaluate predictions across subsets of data: evaluating and CV fitting
- Review and add the a=inf value to the table
- Review SeqDEFT a range values. They don't flatten out at the ends as much as we would like.
- Add some more plotting details to the tutorials e.g. edge coloring and positioning of legends and colorbars in different places
- Make sure every function in the docs is in the API with docstrings as well
- Update plotting binaries with new functions
- Update plotly plotting functionalities
- Review Kernel alignment loss function in the paper, code and docs to match
- Add stationary function distribution to the histogram plot
- Write specific Kernel with uniform noise function that allows direct calculation of the logdet of full covariance matrix that we can use for MLL maximization
- Try to reduce dependencies: seaborn, networkx, datashader. Make some optional: tqdm, fastparquet, plotly
- Add documentation for transition path theory
- Add tests for SeqDEFT intermediates: likelihood calculation under different P's
- Review typos in documentation
- Fix problems in readthedocs updating
- Reorganize extended linear operator: think which methods should be functions to facilate class inheritance and which ones should work with matrices as well
- Think about separation of basic functionalities that are used in visualization and inference
- Review scipy functionality for e.g. cartesian product and compare performance with our implementation and whether we need it at all now. 
- Add bin to rasterize nodes and edges
- Remove test data from repository


### Mid priority
- Reimplement datashader plotting functions using v0.12 native mpl interface
- There is a clear problem when calculating the rate matrix with mutational biases. Review and test if there is still error in calculation of rate matrix when having variable mutation rates. Re-test mutational biases
- Think whether to implement an embedding object
- Streamline reduced alphabet
- Compare inv_dot using minres and taking the reciprocal of the eigenvalues. minres allows having different residual variances per observation though
- Implement the 1SD rule to select hyperparameter in cv
- Better figure out how to calculate the covariance distance using L polynomial and reimplement without L_powers_unique_entries_inv
- Implement product kernel class that uses kron_dot
- Implement kernel alignment on the 2^k different covariances that we can fit as an additive function of rho's and lambda's. The challenge is to extract those values from the data efficiently
  
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