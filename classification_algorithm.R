#load libraries 
library(e1071)
library(randomForest)
library(tree)
library(adabag)
library(caret)
libs <- c("tm", "plyr", "class")
lapply(libs, require, character.only = TRUE)

#set option 
options(stringsAsFactors = FALSE)

#set parameters
doc_type <- c("sport", "tech")
pathname <- "C:/Users/User/Desktop/FIT3164/bbc"

#clean the text using corpus

cleanCorpus <- function(corpus){
  corpus.tmp <- tm_map(corpus, removePunctuation)
  corpus.tmp <- tm_map(corpus.tmp, stripWhitespace)
  corpus.tmp <- tm_map(corpus.tmp, tolower)
  corpus.tmp <- tm_map(corpus.tmp, removeWords, stopwords("english"))
  return(corpus.tmp)
}


#build TDM 
myTDM <- function(doc, path){
  #concatenate doc_type and path
  doc_dir <- sprintf("%s/%s", path, doc)
  
  #create corpus based on the documents inside directory
  doc_corpus <- Corpus(DirSource(directory = doc_dir))
  
  #apply clean corpus function
  doc_clean_cor <- cleanCorpus(doc_corpus)
  
  #create TDM using clean corpus
  doc_tdm <- TermDocumentMatrix(doc_clean_cor)
  
  #remove sparse terms from TDM 
  doc_tdm <- removeSparseTerms(doc_tdm, 0.7)
  
  #create list of result 
  result <- list(name = doc, tdm = doc_tdm)
  
}


tdm <- lapply(doc_type, myTDM, path = pathname)



#attach type of document
bindDocumentToTDM <- function(tdm){
  #convert tdm to data matrix 
  doc_matrix <- t(data.matrix(tdm[["tdm"]]))
  
  #convert to dataframe (each doc in a row, each term in a column, cell=frequency of terms)
  doc_df <- as.data.frame(doc_matrix, stringsAsFactors = FALSE )
  
  #add new column with the name for every rows (where name = type of document) 
  doc_df <- cbind(doc_df, rep(tdm[["name"]], nrow(doc_df)))
  colnames(doc_df)[ncol(doc_df)] <- "targetdocument"
  
  return(doc_df)
}

docTDM <- lapply(tdm, bindDocumentToTDM)



#stack 2 TDM (for tech and sports document)
tdm_stack <- do.call(rbind.fill, docTDM)

#if words does not appear in another doc it will be fill with NA, thus fill it with 0 for TDM
tdm_stack[is.na(tdm_stack)] <- 0


#train-test split 
train_data <- sample(nrow(tdm_stack), ceiling(nrow(tdm_stack) * 0.7))
test_data <- (1:nrow(tdm_stack)) [-train_data]


################ model - KNN ################ 

#extract targetdocument column only 
tdm_doc <- tdm_stack[, "targetdocument"]

#extract other columns 
tdm_stack_ntd <- tdm_stack[, !colnames(tdm_stack) %in% "targetdocument"]

#modelling (p1 = rows without document type to train, p2 = rows without document type to test, p3 = type of document for train data)
knn.pred <- knn(tdm_stack_ntd[train_data, ],  tdm_stack_ntd[test_data, ], tdm_doc[train_data])


#accuracy
confusion_mat <- table("prediction" = knn.pred, "actual" = tdm_doc[test_data])
knn_acc = round(mean(knn.pred == tdm_doc[test_data]) *100, digits = 2)
cat("KNN accuracy is: ", knn_acc, "%") #95.6% 


################ model - Decision Tree ################

#new train and test data
train.row <- sample(1:nrow(tdm_stack), 0.7*nrow(tdm_stack))
new_train_data <- tdm_stack[train.row,]
new_test_data <- tdm_stack[-train.row,]

#Change target variable to factor 
new_train_data["targetdocument"] <- lapply(new_train_data["targetdocument"], factor)
new_test_data$targetdocument = as.factor(new_test_data$targetdocument)

tree.fit <- tree(targetdocument ~., data = new_train_data, method = "class")
tree.pred <- predict(tree.fit, new_test_data, type="class")

tree.cfm = table("actual" = new_test_data$targetdocument, "predicted" = tree.pred)
tree.acc = round(mean(tree.pred == new_test_data$targetdocument)*100, digits = 2)
cat("Decision tree accuracy is: ", tree.acc, "%") #98.18%

################ model - Naive Bayes ################
naive.fit <- naiveBayes(targetdocument ~., data = new_train_data)
naive.pred = predict(naive.fit, new_test_data)
naive.cfm <- table("actual" = new_test_data$targetdocument, "predicted" = naive.pred)
naive.acc <- round(mean(naive.pred == new_test_data$targetdocument)*100, digits=2)
cat("Naïve Bayes model accuracy is: ", naive.acc, "%") #78.1%

################ model - Random Forest ################
RF_model = randomForest(targetdocument ~., new_train_data)
rf.fit.pred <- predict(RF_model, new_test_data)
rf.fit.cfm <- table("actual" = new_test_data$targetdocument, "predicted" = rf.fit.pred)
rf.fit.acc <- round(mean(rf.fit.pred == new_test_data$targetdocument)*100, digits = 2)
cat("Random Forest ensemble model accuracy is: ", rf.fit.acc, "%") #99.64



