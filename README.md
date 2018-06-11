## Popular machine/deep learning algorithms and strategies implemented from scratch using primitive packages, such as numpy and pandas, to make clear understanding of the mathematical concepts behind them. Each algorithm is applied on real-life data 

## Algorithms Implemented

### Supervised Learning
  - [K Nearest Neighbors](Supervised_Learning/KNN.py)
  - [Linear Distriminant Analysis](Supervised_Learning/LinearDiscriminantAnalysis.py)
  - [Linear Regression](Supervised_Learning/LinearRegression.py)
  - [Logistic Regression](Supervised_Learning/LogisticRegression.py)
  - [Naive Bayes](Supervised_Learning/NaiveBayes.py)
  - [Softmax Regression](Supervised_Learning/SoftmaxRegression.py)

### Unsupervised Learning
  - [Fuzzy K Means](Unsupervised_Learning/FuzzyKmeans.py)
  - [K Means / K Medians](Unsupervised_Learning/KMeans_Medians.py)
  - [PCA](Unsupervised_Learning/PCA.py)
  - [Spectral Clustering](Unsupervised_Learning/SpectralClustering.py)

### Deep Learning
  - [Deep Neural Network](Deep_Learning/DeepNeuralNetwork.py)
    - Layers:
      - [Input Layers](Deep_Learning/layers/Input.py)
      - [Dense Layers](Deep_Learning/layers/Dense.py)
      - [Softmax Layers](Deep_Learning/layers/Softmax.py)
    - Optimization Methods:
      - [Stochastic Gradient Descent](Deep_Learning/nn_optimization.py)

### Optimization Methods
  - [Conjugate Gradient Descent](tools/iterative_methods.py)
  - [Jacobi Iteration](tools/iterative_methods.py)
  - [Mini-Batch Gradient Descent](tools/iterative_methods.py)
  - [Steepest Gradient Descent](tools/iterative_methods.py)

## How to use 
  ### Supervised, Unsupervised Learning algorithms work like sci-kit learn:  

  #### Unsupervised:
      fuzzy = FuzzyKmeans(X)
      fuzzy.fit(5) # 5 clusters
      predictions = fuzzy.predict(X) 

  #### Supervised:
      logistic = LogisticRegression(X,y)
      logistic.train(epochs=100)
      predictions = logistic.predict(input)

  ### Deep Learning models are similar to Sequential Models from Keras:
      dnn = DNN(X,y)
      dnn.add("dense","sigmoid",200) # dense layer w/200 neurons and sigmoid activation
      dnn.train(X,y,epochs=100)
      predictions = dnn.predict(input)

### [Examples](Applications/)
