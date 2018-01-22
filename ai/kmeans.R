k_means <- function(frame,k,minimum,maximum) { #takes in a dataframe, number of centers, minimum value and max value
  clusters <- list() #initialize cluster list
  n <- ncol(frame) #number of values
  dim <- length(frame[,1])
  #initialize centers
  for(i in 1:k) {
    clusters[[i]] <- list()
    clusters[[i]][[1]] <- runif(dim,min=minimum,max = maximum) #picks points from uniform distribution
    clusters[[i]][[2]] <- c(0) #initialize vector of points indices
    clusters[[i]][[3]] <- list() #initialize the list of series
  }
  
  check <- TRUE #determines whether to terminate
  clusterindices <- list()
  for(u in 1:k) {
    clusterindices[[u]] <- c(1) #intialize list to check
  }
  count <- 1
  while(check == TRUE) {
    numcheck <- 0
    for(clusts in 1:k) {
      if(all.equal(clusters[[clusts]][[2]],clusterindices[[clusts]]) == TRUE) { #checks to see if we should terminate
        numcheck <- numcheck + 1
      }
      if(numcheck == k) {
        check <- FALSE
      }
      clusterindices[[clusts]] <- clusters[[clusts]][[2]] #update points
      clusters[[clusts]][[2]] <- c(0) #initialize
      clusters[[clusts]][[3]] <- list() #initialize
    }
    #assign to clusters
    for(j in 1:n) {
      series <- frame[,j]
      norms <- c()
      for(centers in 1:k) {
        norm <- norm((clusters[[centers]][[1]] - frame[,j]),"2")
        norms <- c(norms,norm)
      }
      
      cent <- which(norms==min(norms)) #finds closes center to point
      clusters[[cent]][[2]] <- c(clusters[[cent]][[2]],j)
      clusters[[cent]][[3]][[(length(clusters[[cent]][[3]])+1)]] <- series #adds actual points
    }
    for(re in 1:k) {
      clusters[[re]][[2]] <- clusters[[re]][[2]][-1] #remove initial value set
    }
    
    #update centers
    for(z in 1:k) {
      if(length(clusters[[z]][[3]]) >= 1) { #checks if there are more than 0 points in the cluster
        listframe <- data.frame(clusters[[z]][[3]][[1]])
        if(length(clusters[[z]][[3]]) > 1) {
          try(for(listindex in 2:length(clusters[[z]][[3]])) {
            listframe <- cbind(listframe,clusters[[z]][[3]][[listindex]])
          })
        }
        if(length(listframe) > 0) { 
          clusters[[z]][[1]] <- rowMeans(listframe) #calculate the mean of all the points in cluster to get new centers
        }
      }
    }
    count <- count + 1
  }
  return(clusters)
}