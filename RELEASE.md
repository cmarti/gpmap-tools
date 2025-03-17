# Instructions to follow for new release

1. Write down new updates in CHANGELOG
2. Run all tests
```
pytest test
```
3. Create new branch for release [from here](https://nvie.com/posts/a-successful-git-branching-model/)
```
git checkout -b release-x.x dev
```
4. Increase version number in `pyproject.toml` file
5. Commit any additional changes and fixes
6. Merge into master and tag version
```
git checkout master
git merge --no-ff release-x.x # --no-ff keeps separate branches
git tag -a 1.2
```
7. Merge changes into dev branch
```
git checkout dev
git merge --no-ff release-1.2 
```

# Upload to PyPI

1. Create new wheel distribution
2. Test installation from wheel in a new environment and re-run tests
```
conda create -n test_gpmap_release python=3.8.13
conda activate test_gpmap_release
pip install WHEEL
python -m unittest test/test_*py
```
2. Upload to TestPyPI and test installation in a new environment
```
python -m build
twine upload dist/* --repository testpypi
conda create -n test_gpmap_release python=3.8.13
conda activate test_gpmap_release

```

3. Upload to PyPI and test installation in a new environment
```
```
