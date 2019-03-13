#### Classification problem using Keras/Tensorflow
### Predicting if Pima Indians had an onset of diabetes within five years

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from keras.models import Sequential
from keras.layers import Dense
import keras_utilities

### fix random seed for reproducibility
np.random.seed(3)

### Load data
dataset = pd.read_csv("pima-indians-diabetes.csv", delimiter=",").values
### Data has 8 input variables and 1 output
X = dataset[:,0:8]
Y = dataset[:,8]

### Create network
model = Sequential()
### Use rectifier (‘relu‘) activation function on the first layers - only outputs positive values
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
### Use sigmoid function on output layer - ensures output is between 0 and 1, and is easy to assign to a classification or probability
model.add(Dense(1, activation='sigmoid'))

### Compile model
### Use a logarithmic loss function - for a binary classification problem is defined in Keras as “binary_crossentropy“
### Collect and report the classification accuracy as the metric
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

### Fit the model
### "epochs" = Number of iterations through data
### "batch_size" = Number of iterations that are evaluated before an update is made to the weights
model.fit(X, Y, epochs=150, batch_size=10)

### Evaluate the model
scores = model.evaluate(X, Y,verbose=0)
print("{:s}: {:.2f}%".format(model.metrics_names[1], scores[1]*100))

### Save model and weights
keras_utilities.savemodel("model1","weights1")

### Plot loss
pl.plot(model.history.history['loss'],'k-')
pl.xlabel("Number of iterations")
pl.ylabel("Loss")
pl.tight_layout()
pl.show()

### Calculate the predictions
predictions = model.predict(X)
pred_bool = [round(x[0]) for x in predictions]

nsamples = len(Y)
truepositive = (Y - pred_bool) == 0
falsepositive = (Y - pred_bool) == -1
falsenegative = (Y - pred_bool) == 1

ntruepositive = truepositive.sum()
nfalsepositive = falsepositive.sum()
nfalsenegative = falsenegative.sum()

frac_truepositive = ntruepositive/nsamples
frac_falsepositive = nfalsepositive/nsamples
frac_falsenegative = nfalsenegative/nsamples

print("Correctly predicted:\t{0:4d}/{1:d} = {2:6.2f}%\nFalse positives:\t{3:4d}/{1:d} = {4:6.2f}%\nFalse negatives:\t{5:4d}/{1:d} = {6:6.2f}%".format(ntruepositive, nsamples, 100*frac_truepositive, nfalsepositive, 100*frac_falsepositive, nfalsenegative, 100*frac_falsenegative))

have_diabetes = Y.sum()/nsamples
have_no_diabetes = 1.0 - have_diabetes

### Bayes theorm
### Prob(diabetes | positive) = prob(positive|diabetes) * prob(diabetes) / prob(positive)
### Prob(diabetes | positive) = prob(positive|diabetes) * prob(diabetes) / ( prob(positive|diabetes) * prob(diabetes) + prob(positive|no_diabetes) * prob(non_diabetes))
prob_diabetes = frac_correct * have_diabetes / ( frac_correct * have_diabetes + frac_falsepositives * have_no_diabetes)
print("Chance of correct diagnosis: {:6.2f}%".format(100*prob_diabetes))

