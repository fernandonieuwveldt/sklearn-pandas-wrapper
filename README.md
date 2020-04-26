# Wrapper for sklearn transform to return pandas data frame
* Sometimes you want a pandas data frame instead of numpy array return type after applying a transformer. 
* This package contains a simple implementation for such functionality

## To install package:
```
pip install .
```

Imports for the examples

```
import numpy
import pandas
from sklearn_pandas_wrapper import PandasTransformerWrapper, PandasPipelineWrapper, PandasFeatureUnionWrapper
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.datasets import fetch_openml

```
## Example: Wrapping transformer
```
imputer = PandasTransformerWrapper(SimpleImputer, strategy='median')
r = numpy.random.rand(5,3)
r[0, 0] = numpy.nan
df = pandas.DataFrame(r, columns=['a', 'b', 'c'])
print(df)
imputer.fit(df)
imputer.transform(df)

          a         b         c
0       NaN  0.660508  0.713598
1  0.334260  0.469289  0.161776
2  0.952196  0.421329  0.190182
3  0.508684  0.844304  0.916999
4  0.038216  0.167613  0.582439
 
          a         b         c
0  0.421472  0.660508  0.713598
1  0.334260  0.469289  0.161776
2  0.952196  0.421329  0.190182
3  0.508684  0.844304  0.916999
4  0.038216  0.167613  0.582439
```
## Example: a Pipeline
```
# Load data from https://www.openml.org/d/40945
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

categorical_features = ['embarked', 'sex', 'pclass']
categorical_transformer = PandasPipelineWrapper(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

categorical_transformer.fit(X[categorical_features])
print(categorical_transformer.transform(X[categorical_features]))

```

## Example: Feature union
```
steps=[('a', OneHotEncoder()),
       ('b', PolynomialFeatures(2))
       ]
feature_union_data = pandas.DataFrame([[0.5, 1], [0.9, 3], [0.6, 2]], columns=['feature1', 'feature2'])
feature_union = PandasFeatureUnionWrapper(steps)
feature_union.fit(feature_union_data)
feature_union.transform(feature_union_data)

   feature1_0.5  feature1_0.6  feature1_0.9  feature2_1  feature2_2  feature2_3    1  feature1  feature2  feature1^2  feature1 feature2  feature2^2
0           1.0           0.0           0.0         1.0         0.0         0.0  1.0       0.5       1.0        0.25                0.5         1.0
1           0.0           0.0           1.0         0.0         0.0         1.0  1.0       0.9       3.0        0.81                2.7         9.0
2           0.0           1.0           0.0         0.0         1.0         0.0  1.0       0.6       2.0        0.36                1.2         4.0

```
