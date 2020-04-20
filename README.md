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
## Example: a Pipeline can also be wrapped
```
from sklearn import svm 
from sklearn.datasets import make_classification 
from sklearn.feature_selection import SelectKBest 
from sklearn.feature_selection import f_regression 

# generate some data to play with 
X, y = make_classification(n_informative=5, n_redundant=0, random_state=42)
feature_names = [f"feature_{k}" for k in range(20)]
X = pandas.DataFrame(X, columns=feature_names)

# ANOVA SVM-C 
anova_filter = SelectKBest(f_regression, k=5) 
clf = svm.SVC(kernel='linear') 

anova_svm = PandasPipelineWrapper([('anova', anova_filter), ('svc', clf)])
anova_svm.fit(X, y)

print(anova_svm.predict(X))
[1 0 0 1 1 1 0 1 0 0 1 0 1 0 0 1 0 1 0 1 0 1 1 0 0 0 0 1 0 1 0 0 1 1 1 1 1
 0 1 1 1 1 0 0 0 0 0 1 1 0 0 1 0 1 0 0 1 1 1 0 1 0 1 0 0 1 0 1 0 1 1 1 1 0
 1 0 1 1 1 1 0 1 1 0 0 0 0 1 0 0 1 0 1 1 1 0 1 0 1 0]

# get only the transformers of the pipeline
print(anova_svm.transform(X))
    feature_2  feature_3  feature_7  feature_9  feature_11
0   -0.495969   0.415409   0.966621   0.228924    0.882468
1   -1.034373   0.330510   1.566519  -1.640187   -0.796247
2    1.286269   1.062950   1.209124  -2.350830    1.951709
..        ...        ...        ...        ...         ...
97   1.531689   1.257103   1.537580   0.805801    1.386351
98   1.205628   0.552997   0.371916  -0.909731    1.772235
99   1.769117   1.085767  -0.795152  -1.856569    0.295619

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
