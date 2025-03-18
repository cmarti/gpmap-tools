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
git tag -a x.x
```
7. Merge changes into dev branch
```
git checkout dev
git merge --no-ff release-x.x 
```

# Upload to PyPI

1. Create new wheel distribution
2. Test installation from wheel in a new environment and re-run tests
```
conda create -n test_gpmap_release python=3.8
conda activate test_gpmap_release
pip install .
pytest test
```
2. Upload to TestPyPI and test installation in a new environment
```
python -m build
twine upload dist/* --repository testpypi
conda create -n test_gpmap_release python=3.8
conda activate test_gpmap_release
pip install -i https://test.pypi.org/simple/ gpmap-tools==x.x --extra-index-url https://pypi.org/simple/
pytest test 
```

3. Upload to PyPI and test installation in a new environment
```
python3 -m twine upload --repository pypi dist/*
pip install gpmap-tools
pytest test
```
