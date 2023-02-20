## Popular machine/deep learning algorithms and strategies implemented from scratch using primitive packages, such as numpy and pandas, to make clear understanding of the mathematical concepts behind them.

## Algorithms Implemented

### Supervised Learning
  - [K Nearest Neighbors](barebones_ai/supervised/knn.py)
  - [Linear Distriminant Analysis](barebones_ai/supervised/lda.py)
  - [Linear Regression](barebones_ai/supervised/linear_regression.py)
  - [Logistic Regression](barebones_ai/supervised/logistic_regression.py)
  - [Naive Bayes](barebones_ai/supervised/naive_bayes.py)
  - [Softmax Regression](barebones_ai/supervised/softmax_regression.py)

### Unsupervised Learning
  - [Fuzzy K Means](barebones_ai/unsupervised_learning/fuzzy_kmeans.py)
  - [K Means / K Medians](barebones_ai/unsupervised_learning/kmeans.py)
  - [PCA](barebones_ai/unsupervised_learning/pca.py)
  - [Spectral Clustering](barebones_ai/unsupervised_learning/spectral_clustering.py)

### Deep Learning
  - [Deep Neural Network](barebones_ai/supervised/nn/dnn.py)
    - Layers:
      - [Input Layers](barebones_ai/supervised/nn/layers/input.py)
      - [Dense Layers](barebones_ai/supervised/nn/layers/dense.py)
      - [Softmax Layers](barebones_ai/supervised/nn/layers/softmax.py)
    - Optimization Methods:
      - [Stochastic Gradient Descent](barebones_ai/supervised/nn/nn_optimization_methods.py)
  - [AutoEncoder](barebones_ai/supervised/nn/auto_encoder.py)

### Optimization Methods
  - [Conjugate Gradient Descent](barebones_ai/optimization.py)
  - [Jacobi Iteration](barebones_ai/optimization.py)
  - [Mini-Batch Gradient Descent](barebones_ai/optimization.py)
  - [Steepest Gradient Descent](barebones_ai/optimization.py)

## How to use 
  ### Supervised, Unsupervised Learning algorithms work like sci-kit learn:  

  #### Unsupervised:
      # X = input data
      fuzzy = FuzzyKmeans(X)
      fuzzy.fit(5) # 5 clusters
      predictions = fuzzy.predict(X) 

  #### Supervised:
      # X = input data, y = response variables
      logistic = LogisticRegression(X,y)
      logistic.fit(epochs=100)
      predictions = logistic.predict(input)

  ### Deep Learning models are similar to Sequential Models from Keras:
      # X = input data, y = response variables
      dnn = DNN()
      dnn.add(Input(X))
      dnn.add(Dense(200,"sigmoid")) # dense layer w/ 200 neurons and sigmoid activation
      dnn.add(Softmax(10)) # Softmax layer mapping to 10 classes
      dnn.fit(X,y,lr=0.001,epochs=100) #train for 100 epochs w/ a learning rate of 0.001
      predictions = dnn.predict(input)

### [Examples](notebooks/)
