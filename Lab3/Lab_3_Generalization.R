###################################################################3
# Lab 3: Bayesian models of inductive generalization
# This lab has been developed from a lab originally made by Perfors and Navarro.
# It recreates the models in Sanjana & Tenenbaum
############################################################

##############################################################
# PART 1;   A VERY SIMPLE GENERALISATION
##############################################################
# Creating simple generalizations can be understood best if you think of it
# as having two parts: First...create the hypothesis space, then calculate the priors (belief)
# and the likelihood,e tc. We'll first do it for a very simple example, and then you can 
# complete it for the actual Sanjana and Tenenbaum study. 
# You may have to clear the memory fairly frequently:
rm(list=ls())

#############################################3
#STEP 1: 
# We are going to create a graph with two leaves, horse and cow, that are joined together
# in a new node, calling the top node "hors&cow" 
##############################################

items <- c("horse", "cow")

tree <- rbind(
  c(1,0), # all the singleton clusters
  c(0,1),
  c(1,1) # all the pair clusters
)

colnames(tree) <- items # attach nice labels

# The dimension of the cluster should be 2 by 3
nClusters <- dim(tree)[1] 
nItems <- length(items)

# What is the hypothesis space. 
# Choose show syou how many sets of n-elements can you choose from a set with k-elements
# choose(k,n)
# So, since nClusters is 3, (horse, cow, or horse&cow)
# So that means 3 singleton sets (1 of the 1:2) = 3
# ANd then there are also 3 sets of 2, e.g. horse + cow, horse + (horse&cow), cow + (horse&cow) 
# You can see this by checking choose(3,1) and choose (3,2)
# Add these up to get the number of hypothesis we could possibly have

nHypotheses <- sum( choose( nClusters, 1:2 )) # <- upper bound!

# initialise a hypothesis space that consists of all possible
# clusters and pairs of clusters. Create a belief vector 
# where the prior probabilities can be stored

#This says create a 3 by 2 matrix, where every value is initialized to 0
hypotheses <- matrix( 0, nrow=nHypotheses, ncol=nItems )

colnames( hypotheses ) <- items
# initialize a belief vector (this is going to be the prior probablity)
belief <- vector( length=nHypotheses )
#Initially we know no values, so each hypothesis is false

# the first order hypotheses are just the 3 clusters defined 
# by tree structure itself
# We are going to put these into the belief structure
hypotheses[ 1:nClusters, ] <- tree
# Phi is a normalizing value. 
phi = 20 # You can check what change this value does
belief[ 1:nClusters ] <- 1/phi
# Check what belief looks like now
belief


# This adds hypothesis space for pairs of clusters
ind <- nClusters
for( a in 1:(nClusters-1) ) {
  for( b in (a+1):nClusters ) {
    
    ind <- ind+1
    hypotheses[ind, ] <- tree[a,] | tree[b,] 
    belief[ind] <- (1/phi)^2
    
  }
}



# However, many sets are identical 
# (and many more will be when we start using bigger graphs.)
# for instance, there's 
# a cluster for "horse" and a cluster for "cow" in the tree
# and there's also a cluster for "horse,cow". so there's no
# reason to include "horse"+"cow" as a composite hypothesis,
# because it's already there as a first order one. 
#
# to fix this, we'll remove all duplicated rows from the 
# hypothesis space. specifically, because the hypotheses
# are ordered (first order at the top, third order at the
# bottom), what we want to do is keep the *FIRST* instance
# of a particular row. 
#
# the R function duplicated is perfect for this. It picks 
# out all of the rows that are duplicates of rows above it

belief
sum(belief)
#(What does the sum add up to now? )

redundant <- duplicated( hypotheses, MARGIN=1 )

# keep only the non-redundant hypotheses, and then normalise
# the belief vector so that it sums to 1
hypotheses <- hypotheses[!redundant,]
# Look at what belief is.
belief
# Now remove the reundant items and check again
belief <- belief[!redundant]
belief
sum(belief)
# But we have a problem, if you 
# Now that the redundant items are removed, we need to normalize again
belief <- belief / sum( belief )
sum(belief)
nHypotheses <- length( belief ) # <- actual number of hypotheses
nHypotheses
belief
hypotheses

##################################################
# STEP 2
# We have 3 hypotheses, one with just cow, one with just horse, and one with both
# We have prior probablities for each of the three possibilites
# (they are equally probable)
###############################################
# Now we have to calculate the likelihood (given )
# First initialize a 2 x 3 matrix with 0's in each position
likelihood <- matrix( 0, nrow=nHypotheses, ncol=nItems)
colnames(likelihood) <- items
# for each of the 3 rows in the hypothesis, there are two values
# 1 0
# 0 1
# 1 1
# We want to calculate the probability of horse and cow if we are in that hypothesis space
# where the first row is hypothesis space where only the singleton horse belongs, the second row is the singleton cow
# and the third row is the node horse&cow
# The calculation is simple for the first two rows, 1/1 = 1.0 and 0/1 is 0 
# For the third hypothesis, 1/2 = 0.5, and 1/2 = 0.5 
# Note that each row adds up to 1, as expected

for(x in 1:nHypotheses ) {
  likelihood[x, hypotheses[x,]==1 ] <- 1/sum(hypotheses[x,])  
}

# Look at the likelihood matrix. You can the probability of the different hypotheses. Compare this to the
# hypotheses, 
likelihood

# Now we are going to update our beliefs given the likelihoods for a given premise
# First define the premise
# This is the set of animals the subjects were told had the disease, e.g.
premises = ("cow")
# premises =("horse")
# premises =("cow","cow")
for( x in premises ) {
  belief <- belief * likelihood[,x]
}

# The sum of the belief is not 1...need to normalize
belief <- belief / sum(belief) # must sum to 1
belief
sum(belief)

# Now compute the generalisation probabilities using 
# matrix multiplication: multiply the belief vector by the 
# hypothesis matrix...
# Perhaps first check what the blief and hypotheses vectors are so you can understand what happens
hypotheses

generalisations <- belief %*% hypotheses

# Now you can look at the generalisations and see for a given animal, what the model predicts
# the likelihood is that that animal gets the disease, given that the animals in the 
# premise have the disease. 
generalisations

# You can also create a barplot. 
barplot( generalisations, ylab="generalisation probability", las=2,
         main=paste( "premises:", paste(premises,collapse=",")),
         font.main=1 )


###############################################
# PART 2: Recreating Sanjana& Tanenbaum
############################
  
# names of the animals
items <- c("horse", "cow", "elephant", "rhino", "chimp", 
               "gorilla", "mouse", "squirrel","dolphin", "seal" )
  
}  

  # The "base" representation is a simple binary tree structure.
  # In the original work, Sanjana & Tenenbaum derived this tree
  # by applying a hierarchical clustering algorithm to human 
  # similarity judgments. 
  tree <- rbind(
    
    c(1,0,0,0,0,0,0,0,0,0), # all the singleton clusters
    c(0,1,0,0,0,0,0,0,0,0),
    c(0,0,1,0,0,0,0,0,0,0),
    c(0,0,0,1,0,0,0,0,0,0),
    c(0,0,0,0,1,0,0,0,0,0),
    c(0,0,0,0,0,1,0,0,0,0),
    c(0,0,0,0,0,0,1,0,0,0),
    c(0,0,0,0,0,0,0,1,0,0),
    c(0,0,0,0,0,0,0,0,1,0),
    c(0,0,0,0,0,0,0,0,0,1),
    
    c(1,1,0,0,0,0,0,0,0,0), # all the pairs in the tree
    c(0,0,1,1,0,0,0,0,0,0),
    c(0,0,0,0,1,1,0,0,0,0),
    c(0,0,0,0,0,0,1,1,0,0),
    c(0,0,0,0,0,0,0,0,1,1),
    
    c(1,1,1,1,0,0,0,0,0,0), # the bigger ones
    c(1,1,1,1,1,1,0,0,0,0),
    c(1,1,1,1,1,1,1,1,0,0),
    c(1,1,1,1,1,1,1,1,1,1)
    
  )

  colnames(tree) <- items # attach nice labels
  
  
  # useful numbers
  nClusters <- dim(tree)[1] 
  nItems <- length(items)
  nHypotheses <- sum( choose( nClusters, 1:3 )) # <- upper bound!
  
  
  # initialise a hypothesis space that consists of all possible
  # clusters, pairs of clusters, and triples of clusters. also
  # a belief vector that describes our prior over these clusters
  hypotheses <- matrix( 0, nrow=nHypotheses, ncol=nItems )
  colnames( hypotheses ) <- items
  belief <- vector( length=nHypotheses )
  
  # the first order hypotheses are just the 19 clusters defined 
  # by tree structure itself
  hypotheses[ 1:nClusters, ] <- tree
  phi = 20
  belief[ 1:nClusters ] <- 1/phi
  
  # the second order hypotheses are unique pairs of 
  # clusters in the tree (or, as it will turn out, a subset
  # of these pairs)
  ind <- nClusters
  for( a in 1:(nClusters-1) ) {
    for( b in (a+1):nClusters ) {
      
      ind <- ind+1
      hypotheses[ind, ] <- tree[a,] | tree[b,] 
      belief[ind] <- (1/phi)^2
      
    }
  }
  
  # the third order hypotheses are the unique triples of 
  # clusters in the tree (or, as it will turn out, a subset
  # of these pairs)
  for( a in 1:(nClusters-2) ) {
    for( b in (a+1):(nClusters-1) ) {
      for( c in (b+1):nClusters ) {
        
        ind <- ind+1
        hypotheses[ind, ] <- tree[a,] | tree[b,] | tree[c,]
        belief[ind] <- (1/phi)^3
              
      }  
    }
  }
  

  # Now we'll want tpo remove the all duplicated rows from the 
  # hypothesis space. specifically, because the hypotheses
  # are ordered (first order at the top, third order at the
  # bottom), what we want to do is keep the *FIRST* instance
  # of a particular row. 
  #
  # the R function duplicated picks 
  # out all of the rows that are duplicates of rows above it
  redundant <- duplicated( hypotheses, MARGIN=1 )
  
  # keep only the non-redundant hypotheses, and then normalise
  # the belief vector so that it sums to 1
  hypotheses <- hypotheses[!redundant,]
  belief <- belief[!redundant]
  belief <- belief / sum( belief )
  nHypotheses <- length( belief ) # <- actual number of hypotheses
  
  # create likelihoods
  likelihood <- matrix( 0, nrow=nHypotheses, ncol=nItems)
  colnames(likelihood) <- items
  for( ind in 1:nHypotheses ) {
    likelihood[ind, hypotheses[ind,]==1 ] <- 1/sum(hypotheses[ind,])  
  }
  
  # now show the model the data, and sequentially update beliefs
  # HEre's you'll also want to define your premises:
  premises = ("dolphin")
  #premises = ("dolphin","dolphin","dolphin")

  for( x in premises ) {
    belief <- belief * likelihood[,x]
  }

    belief <- belief / sum(belief) # must sum to 1
  
  # now compute the generalisation probabilities. I could do this
  # with loops, but it actually corresponds to a really simple
  # matrix multiplication: multiply the belief vector by the 
  # hypothesis matrix...
  generalisations <- belief %*% hypotheses
  

  
  barplot( generalisations, ylab="generalisation probability", las=2,
         main=paste( "premises:", paste(premises,collapse=",")),
         font.main=1 )

