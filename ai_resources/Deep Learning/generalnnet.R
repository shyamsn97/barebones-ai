
#@author Shyam Sudhakaran
#This program uses a multi-layered neural network to approximate a function(in this case, sin(x))

library(stats)
library(ggplot2)


y_function <- function(x) {
  sin(x)
}

create_var <- function(dimensions,funct,range,length) {
  x <- matrix(runif(length,-1*range,range),nrow = dimensions)
  y <- funct(x)
  return(list(x,y))
}

variables <- create_var(1,y_function,10,100)
node_levels <- c(nrow(variables[[1]]),30,40,20,1)
constants_generator <- function(node_layers,range) {
constants <- list()
  for(i in 1:(length(node_layers)-1)) {
    constants[[i]] <- matrix(runif(node_layers[[i]]*node_layers[[i+1]],-1*range,range),ncol = node_layers[[i]],nrow = node_layers[[i+1]]) 
  }
  return(constants)
}
constants <- constants_generator(node_levels,1)

hiddenfunc <- function(...) {
  #return(1/(1 + exp(-1*input)))
  return(log(1+exp(...)))
}
derivhiddenfunc <- function(...) {
  #return((1/(1 + exp(-1*input)))*(1-(1/(1 + exp(-1*input)))))
  return(1/(1 + exp(-1*...)))
}

foward_propagate <- function(constants_,xval,func) {
  vec <- xval
  layerlist <- list()
  for(i in 1:length(constants_)) {
    if(i == 1) {
      vec <- constants_[[i]]%*%xval
      layerlist[[i]] <- func(vec)
    } 
    else if(i > 1 && i < length(constants_)) {
      vec <- constants_[[i]]%*%layerlist[[i-1]]
      layerlist[[i]] <- func(vec)
    }
    else {
      vec <- constants_[[i]]%*%layerlist[[i-1]]
      layerlist[[i]] <- vec
    }
  }
  return(layerlist)
}

deriv_propagate <- function(layer,const,xval,func,derivfunc) {
  vec <- xval
  devlist <- list()
  devlist[[length(const)]] <- layer[[length(layer)-1]]
  for(i in 1:length(const)) {
    if(i == 1) {
      vec <- const[[i]]%*%xval
      devlist[[i]] <- derivfunc(vec)
    } 
    else if(i > 1 && i < length(const)) {
      vec <- const[[i]]%*%layer[[i-1]]
      devlist[[i]] <- derivfunc(vec)
    }
  }
  return(devlist)
}

nnylist <- foward_propagate(constants,variables[[1]],hiddenfunc)
nny <- nnylist[[length(nnylist)]]
nnylistderiv <- deriv_propagate(nnylist,constants,variables[[1]],hiddenfunc,derivhiddenfunc)
nnylistderiv <- nnylistderiv[1:(length(nnylistderiv)-1)]
layerlist <- nnylist[1:(length(nnylist)-1)]


calculate_gradient <- function(layers,derivlayers,x,constants,y,nny) {
  chains <- list()
  deriv <- list()
  chains[[length(constants)]] <- (nny-y)
  deriv[[length(constants)]] <- (nny-y)%*%t(layers[[length(layers)]])
  for(i in length(layers):1) {
    chains[[i]] <- (t(constants[[i+1]])%*%chains[[i+1]])*derivlayers[[i]]
    if(i == 1) {
      deriv[[i]] <- chains[[i]]%*%t(x)
    }
    else {
      deriv[[i]] <- chains[[i]]%*%t(layers[[i-1]])
    }
  }
  return(deriv)
}

gradients <- calculate_gradient(layerlist,nnylistderiv,variables[[1]],constants,variables[[2]],nny)

grad_descent <- function(layer,listderiv,gradient,alpha,constantlist,output,var) {
  MSE <- 0.5*sum((var[[2]]-output)^2)
  count <- 0
  prevMSE <- MSE + 2
  while(MSE>0.1 && (abs(prevMSE-MSE) > 0.00000001)) {
    prevMSE <- MSE
    count <- count + 1
    for(i in 1: length(constantlist)) {
      constantlist[[i]] <- constantlist[[i]] - (alpha * gradient[[i]])
    }
    out <- foward_propagate(constantlist,var[[1]],hiddenfunc)
    layer <- out[1:(length(out)-1)]
    out <- out[[length(out)]]
    listderiv <- deriv_propagate(layer,constantlist,var[[1]],hiddenfunc,derivhiddenfunc)
    listderiv <- listderiv[1:(length(listderiv)-1)]
    gradient <- calculate_gradient(layer,listderiv,var[[1]],constantlist,var[[2]],out)
    MSE <- 0.5*sum((var[[2]]-out)^2)
    print(prevMSE)
    print(MSE)
    print(count)
  }
  return(out)
}

o <- grad_descent(layerlist,nnylistderiv,gradients,0.000005,constants,nny,variables)

plot(variables[[1]],variables[[2]],col = "blue", pch = 20) + points(variable[[1]],o,col = "red", pch = 20)

