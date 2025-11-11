### High priority
- Implement posterior covariance under MEI (requires 'a').
- API section: need to have plotting functions and docstrings for DataSet methods
- Write __str__ method for DataSet to highlight the main properties of the dataset
- Make clean and complete datasets
- Add data file with citations for the built-in datasets that is updated if new datasets are added with reference
- Fix mixing time notation to relaxation and write the neutral relaxation time method for cases with mutational biases
- Add some more plotting details to the tutorials e.g. edge coloring and positioning of legends and colorbars in different places
- Make sure every function in the docs is in the API with docstrings as well
- Review Kernel alignment loss function in the paper, code and docs to match

### Mid priority
- Re-implement HMC
- Try to use datashader only to generate the bitmap and plot it with matplotlib
- Reimplement datashader plotting functions using v0.12 native mpl interface
- Add functionality to store and read rasterized nodes and edges using datashader
- Review SeqDEFT a range values. They don't flatten out at the ends as much as we would like.
- Explore whether VC regression is faster using the cost matrix implementation
- Try to reduce dependencies: networkx, datashader. Make some optional: tqdm, pyarraw, plotly
- Fix and complete documentation about evolution on fitness landscapes
- Add documentation for transition path theory
- Change "a" parametrization to sigma or variance
- There is a clear problem when calculating the rate matrix with mutational biases. Review and test if there is still error in calculation of rate matrix when having variable mutation rates. Re-test mutational biases
- Streamline reduced alphabet
- Implement the 1SD rule to select hyperparameter in cv
  
### Low priority
- Implement new likelihood functions: poisson and binomial at least, maybe something similar to GLM for regression like approaches: have likelihood function classes with log-likelihood and gradient methods. Another interesting possiblity is the use of contrastive losses
- Create new classes of Discrete spaces for the extra models to generate visualizations from: smily face and tree-like
- Write a method to plot a phylogenetic tree on a landscape: test on the codon landscape
- Write a method to simulate evolution in the landscape: use the jump matrix and exponential waiting times. 
- Write a method to simulate sequence evolution on a phylogenetic tree using the previous function.
- Go back and review polynomial coefficients calculations precision with different methods. We lose precision with l=7 or 8 already with current implementation
- Figure out inconsistency between using the correct eigenvalues and getting wrong projections in the skewed kernel. So far it makes more sense to have the correct projection operators
- Figure out matrix free versions of Skewed Laplacian

### Long term
- Implement evidence maximization
- Implement contrastive losses
- Add section to docs on the practical implementation of the operators as compared with traditional
  gaussian process approaches that use the dense Cholesky decomposition, which is intractable for 
  large datasets.
- Try MCMC on marginal likelihood
- Add star tree phylogenetic likelihood