# Wrapper for sklearn transform to return pandas data frame instead of a numpy array
* Sometimes you want a pandas data frame instead of numpy array return type after applying a transformer. 
* This package contains a simple implementation for such functionality

## Example
```
import numpy
import pandas
from sklearn.impute import SimpleImputer
from sklearn_pandas_wrapper import PandasTransformerWrapper
# wrap sklearn transformer
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

### Not yet implemented:
Todo: Check for special cases where number of features change after transform
      For example OneHotEncoder
Todo: Add functionality for pipelines
