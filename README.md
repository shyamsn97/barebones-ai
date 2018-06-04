## Popular machine/deep learning algorithms and strategies implemented from scratch using numpy and applied on real-life data 

## Algorithms Implemented

### Supervised Learning
  - [K Nearest Neighbors](Supervised Learning/KNN.py)
  - [Linear Distriminant Analysis](Supervised Learning/LinearDiscriminantAnalysis.py)
  - [Linear Regression](Supervised Learning/LinearRegression.py)
  - [Logistic Regression(]Supervised Learning/LogisticRegression.py)
  - [Naive Bayes](Supervised Learning/NaiveBayes.py)

### Unsupervised Learning
  - [Fuzzy K Means](Unsupervised Learning/FuzzyKmeans.py)
  - [K Means / K Medians](Unsupervised Learning/KMeans_Medians.py)
  - [PCA](Unsupervised Learning/PCA.py)
  - [Spectral Clustering](Unsupervised Learning/SpectralClustering.py)

### Deep Learning
  - [Deep Neural Network](Deep Learning/DeepNeuralNetwork.py)
    - Layers:
      - [Input Layers](Deep Learning/layers/Input.py)
      - [Dense Layers](Deep Learning/layers/Dense.py)

### Optimization Methods
  - [Conjugate Gradient Descent](tools/iterative_methods.py)
  - [Jacobi Iteration](tools/iterative_methods.py)
  - [Mini-Batch Gradient Descent](tools/iterative_methods.py)
  - [Steepest Gradient Descent](tools/iterative_methods.py)

## How to use 
  ### Supervised, Unsupervised Learning algorithms work like sci-kit learn:  

  #### Unsupervised:
      fuzzy = FuzzyKmeans(X)
      predictions = fuzzy.predict(5) # 5 clusters

  #### Supervised:
      logistic = LogisticRegression(X,y)
      logistic.train()
      predictions = logistic.predict(input)

  ### Deep Learning models are similar to Sequential Models from Keras:
      dnn = DNN(X,y)
      dnn.add("dense","sigmoid",200) # dense layer w/200 neurons and sigmoid activation
      dnn.train(X,y)
      predictions = dnn.predict(input)

## [Examples](Applications/)
