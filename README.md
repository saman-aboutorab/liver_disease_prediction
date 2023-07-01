# Liver Disease prediction
:::

::: {.cell .markdown id="xXvypLxq1m3F"}
## Introduction

In the following project, we\'ll work with the Indian Liver Patient
Dataset from the UCI Machine learning repository.

We\'ll instantiate three classifiers to predict whether a patient
suffers from a liver disease using all the features present in the
dataset.
:::

::: {.cell .markdown id="ix81J47s2EFO"}
![liver
disease](vertopal_2fe58ac8e73340d08b10cb9b596c7be8/da1435fdf5146c298c2482a9ce93f3ad7b094bd4.jpg)
:::

::: {.cell .code execution_count="36" id="zVSyWTWz0sP9"}
``` python
from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier as KNN

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
```
:::

::: {.cell .markdown id="ncE70_Hk3x7A"}
## Dataset
:::

::: {.cell .code execution_count="9" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="OdSPlMB53zbe" outputId="fcb5ab81-b0c6-4899-ef4c-cf1468356354"}
``` python
df_patients = pd.read_csv('indian_liver_patient_preprocessed.csv')
print(df_patients.head())
```

::: {.output .stream .stdout}
       Unnamed: 0   Age_std  Total_Bilirubin_std  Direct_Bilirubin_std  \
    0           0  1.247403            -0.420320             -0.495414   
    1           1  1.062306             1.218936              1.423518   
    2           2  1.062306             0.640375              0.926017   
    3           3  0.815511            -0.372106             -0.388807   
    4           4  1.679294             0.093956              0.179766   

       Alkaline_Phosphotase_std  Alamine_Aminotransferase_std  \
    0                 -0.428870                     -0.355832   
    1                  1.675083                     -0.093573   
    2                  0.816243                     -0.115428   
    3                 -0.449416                     -0.366760   
    4                 -0.395996                     -0.295731   

       Aspartate_Aminotransferase_std  Total_Protiens_std  Albumin_std  \
    0                       -0.319111            0.293722     0.203446   
    1                       -0.035962            0.939655     0.077462   
    2                       -0.146459            0.478274     0.203446   
    3                       -0.312205            0.293722     0.329431   
    4                       -0.177537            0.755102    -0.930414   

       Albumin_and_Globulin_Ratio_std  Is_male_std  Liver_disease  
    0                       -0.147390            0              1  
    1                       -0.648461            1              1  
    2                       -0.178707            1              1  
    3                        0.165780            1              1  
    4                       -1.713237            1              1  
:::
:::

::: {.cell .code execution_count="10" id="EEgHVQ9C4GSo"}
``` python
df_patients = df_patients.drop(['Unnamed: 0'], axis=1)
```
:::

::: {.cell .markdown id="GGR8nLg_4zbs"}
## Train/Test split
:::

::: {.cell .code execution_count="13" id="T4-HC-_14M3K"}
``` python
X = df_patients.drop(['Liver_disease'], axis=1)
y = df_patients[['Liver_disease']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
```
:::

::: {.cell .markdown id="gKcWOHNn30BD"}
## Define classifier models
:::

::: {.cell .code execution_count="3" id="-E7neQbI2kHs"}
``` python
SEED = 1
# Instantiate lr
lr = LogisticRegression(random_state = SEED)

# Instantiate KNN
knn = KNN(n_neighbors=27)

# Instantiate dr
dt = DecisionTreeClassifier(min_samples_leaf=0.13, random_state=SEED)

# List of classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbors', knn), ('Classification Tree', dt)]
```
:::

::: {.cell .markdown id="ts9v1yEW463C"}
## Evaluate Classifiers
:::

::: {.cell .code execution_count="17" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="Lyg-5k513DqJ" outputId="efbffccc-baa9-4fb1-ebf9-dfefd03c1c55"}
``` python
# Iterate over classifier
for clf_name, clf in classifiers:

  # Fit to the training data
  clf.fit(X_train, y_train)

  # Predict test data
  y_pred = clf.predict(X_test)

  # Calculate accuracy
  accuracy = accuracy_score(y_test, y_pred)

  # Evaluate clf's accuracy on the test set
  print('{:s} : {:.3f}'.format(clf_name, accuracy))
```

::: {.output .stream .stdout}
    Logistic Regression : 0.690
    K Nearest Neighbors : 0.698
    Classification Tree : 0.672
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py:1143: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /usr/local/lib/python3.10/dist-packages/sklearn/neighbors/_classification.py:215: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
      return self._fit(X, y)
:::
:::

::: {.cell .markdown id="nD5gVQfq6GaU"}
## Voting classifiers
:::

::: {.cell .code execution_count="20" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="L8UyaycS4_lC" outputId="cff173e0-a042-4955-86c8-ae8ad020bc48"}
``` python
# Instantiate a VottingClassifier
vc = VotingClassifier(estimators = classifiers)

# Fit to training data
vc.fit(X_train, y_train)

# predict
y_pred = vc.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print('Voting Classifer: {:.3f}'.format(accuracy))
```

::: {.output .stream .stdout}
    Voting Classifer: 0.681
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_label.py:99: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_label.py:134: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, dtype=self.classes_.dtype, warn=True)
:::
:::

::: {.cell .markdown id="ypcfLm-z7jqp"}
## Bagging classifier
:::

::: {.cell .code execution_count="26" id="h7eDBuwL6yBm"}
``` python
# Instantiate dt
dt = DecisionTreeClassifier(random_state=1)

# Instantiate bc
bc = BaggingClassifier(base_estimator=dt, n_estimators=50, oob_score=True, random_state=1)
```
:::

::: {.cell .code execution_count="27" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="L91uJ-3z70xR" outputId="9d4028f2-edab-444b-ca8e-11405c40e42c"}
``` python
# Fit bc on training data
bc.fit(X_train, y_train)

# Predict bc on test data
bc.predict(X_test)

# Evaluate
acc_test = accuracy_score(y_test, y_pred)

# Evaluate OOB accuracy
acc_oob = bc.oob_score_

# Print acc_test and acc_oob
print('Test set accuracy: {:.3f}, OOB accuracy: {:.3f}'.format(acc_test, acc_oob))
```

::: {.output .stream .stdout}
    Test set accuracy: 0.681, OOB accuracy: 0.737
:::

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_bagging.py:802: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_base.py:166: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.
      warnings.warn(
:::
:::

::: {.cell .markdown id="bowJN4gXAnlM"}
## Adaboost classifier
:::

::: {.cell .code execution_count="33" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="uXQ27l1SAnWd" outputId="0641f78c-5655-4cb5-d461-eec34958d74a"}
``` python
# Instantiate ada
ada = AdaBoostClassifier(base_estimator=dt, n_estimators=180, random_state=1)

# Fit to the training data
ada.fit(X_train, y_train)

# Predict on test data
y_pred_proba = ada.predict_proba(X_test)[:,1]
```

::: {.output .stream .stderr}
    /usr/local/lib/python3.10/dist-packages/sklearn/utils/validation.py:1143: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
      y = column_or_1d(y, warn=True)
    /usr/local/lib/python3.10/dist-packages/sklearn/ensemble/_base.py:166: FutureWarning: `base_estimator` was renamed to `estimator` in version 1.2 and will be removed in 1.4.
      warnings.warn(
:::
:::

::: {.cell .markdown id="jV-xGVCIBeaD"}
## Evaluate ada
:::

::: {.cell .code execution_count="35" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="iA0cdJHh8OJp" outputId="496e871b-c00c-4d9c-995f-0dd9106d86f3"}
``` python
# Evaluate ada classifier
ada_roc_auc_score = roc_auc_score(y_test, y_pred_proba)

print('ROC AUC score: {:.3f}'.format(ada_roc_auc_score))
```

::: {.output .stream .stdout}
    ROC AUC score: 0.657
:::
:::

::: {.cell .markdown id="rctC7iBoDX4a"}
## GridSearchCV: Decision Tree hyperparamets
:::

::: {.cell .code execution_count="43" colab="{\"base_uri\":\"https://localhost:8080/\",\"height\":118}" id="ImRISWv3BzRT" outputId="d5b1cfeb-f0b0-485a-9d66-54ef8c1b2127"}
``` python
# Define params
params_dt = {'max_depth':[2, 3, 4], 'min_samples_leaf':[0.12, 0.14, 0.16, 0.18]}

# Instantiate grid
grid_dt = GridSearchCV(estimator=dt, param_grid=params_dt, scoring='roc_auc', cv=5, n_jobs=-1)

grid_dt.fit(X_train, y_train)
```

::: {.output .execute_result execution_count="43"}
```{=html}
<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5, estimator=DecisionTreeClassifier(random_state=1), n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [2, 3, 4],
                         &#x27;min_samples_leaf&#x27;: [0.12, 0.14, 0.16, 0.18]},
             scoring=&#x27;roc_auc&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5, estimator=DecisionTreeClassifier(random_state=1), n_jobs=-1,
             param_grid={&#x27;max_depth&#x27;: [2, 3, 4],
                         &#x27;min_samples_leaf&#x27;: [0.12, 0.14, 0.16, 0.18]},
             scoring=&#x27;roc_auc&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(random_state=1)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(random_state=1)</pre></div></div></div></div></div></div></div></div></div></div>
```
:::
:::

::: {.cell .markdown id="NppQReDEEtXo"}
## GridSearch result
:::

::: {.cell .code execution_count="44" colab="{\"base_uri\":\"https://localhost:8080/\"}" id="2R0eyw_mEkhP" outputId="6c77844e-682a-4bc8-dad8-c0a50a564cda"}
``` python
# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))
```

::: {.output .stream .stdout}
    Test set ROC AUC score: 0.696
:::
:::

::: {.cell .code id="8dL5KqhoFJEM"}
``` python
```
:::
