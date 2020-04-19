from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()


setup(
    name="sklearn_pandas_wrapper",
    version="0.1.0",
    author="Fernando Nieuwveldt",
    author_email="fdnieuwveldt@gmail.com",
    description="A package to return data frame format from sklearn transformers",
    long_description=readme,
    url="https://github.com/fernandonieuwveldt/sklearn-pandas-wrapper",
    packages=find_packages(),
)
