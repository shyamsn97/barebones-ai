## Popular machine/deep learning algorithms and strategies implemented from scratch using numpy and applied on real-life data 

## Algorithms Implemented

### Supervised Learning
  - K Nearest Neighbors
  - Linear Distriminant Analysis
  - Linear Regression
  - Logistic Regression
  - Naive Bayes

### Unsupervised Learning
  - Fuzzy K Means
  - K Means / K Medians
  - PCA
  - Spectral Clustering

### Deep Learning
  - Deep Neural Network
    - Layers:
      - Input Layers
      - Dense Layers

### Optimization Methods
  - Conjugate Gradient Descent
  - Jacobi Iteration
  - Mini-Batch Gradient Descent
  - Steepest Gradient Descent

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
