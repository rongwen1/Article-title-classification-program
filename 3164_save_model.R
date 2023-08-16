rm(list = ls())
library("slam")
library("tm")
library("SnowballC")
library("dplyr")
library("XML")
library("xml2")
library("RCurl")
library("e1071")
library("randomForest")
library("tree")
library("adabag")
library("caret")
library("plyr")
library(caTools)
library(neuralnet)
library(ROCR)


set.seed(10000)

file_list <- list.files()
articles_table <- data.frame(matrix(ncol = 4, nrow = 0))

for (article in file_list[1:2000]) {
  # read an article
  article <- read_xml(article,'UTF_8')
  
  
  title <- xml_text(xml_find_all(article, xpath = "//title-group"))
  abstract <- xml_text(xml_find_all(article, xpath = "//abstract"))
  body <- xml_text(xml_find_all(article, xpath = "//body"))
  category <- xml_text(xml_find_all(article, xpath = "//article-categories"))
  
  clean <- function(x){
    
    x <-removeNumbers(x)
    
    x <-removePunctuation(x)
    
    x <-tolower(x)
    
    x <-removeWords(x,stopwords('en'))
    
    x <-stripWhitespace(x)
    
    x <-stemDocument(x)
    
    return(x) }
  
  
  title <- clean(title)
  abstract <- clean(abstract)
  body <- clean(body)
  category <- clean(category)
  
  temp_abs <- ""
  if (length(abstract) > 1){
    for (text in abstract){
      temp_abs <- paste(temp_abs,text)
    }
  }
  abstract <- temp_abs
  
  articles_table <- rbind(articles_table, c(title,abstract,body,category))
  
  
}

colnames(articles_table) = c("Title","Abstract","Paragraphs","Category")


#################################Create term document matrix########################################################

#Abstract TDM
myCorpus <- Corpus(VectorSource(articles_table$Abstract))

tdm_abstract <- TermDocumentMatrix(myCorpus)

tdm_abstract = as.data.frame(as.matrix(tdm_abstract))

tdm_abstract = t(tdm_abstract)

tdm_abstract = as.data.frame(tdm_abstract, stringsAsFactors = FALSE)


#Paragraph TDM
paraCorpus <- Corpus(VectorSource(articles_table$Paragraphs))

tdm_paragraph <- TermDocumentMatrix(paraCorpus)

tdm_paragraph <- removeSparseTerms(tdm_paragraph, 0.7)

tdm_paragraph = as.data.frame(as.matrix(tdm_paragraph))

tdm_paragraph = t(tdm_paragraph)

tdm_paragraph = as.data.frame(tdm_paragraph, stringsAsFactors = FALSE)


articles_table$abstractTotal <- rowSums(as.matrix(tdm_abstract))
articles_table$paragraphTotal <- rowSums(as.matrix(tdm_paragraph))
articles_table$abstractDensity <- NA
articles_table$paragraphDensity <- NA

abstractCount = 0 
paragraphCount = 0

for (i in 1:length(articles_table$Title)){
  word_list = as.list(scan(text=articles_table$Title[i], what="[[:space:]]"))
  for ( words in word_list) {
    if ( ! is.null(tdm_abstract[i,words]) ){
      if (tdm_abstract[i,words] > 0 ){
        abstractCount = abstractCount + tdm_abstract[i,words]
        absDensity = abstractCount / articles_table$abstractTotal[i]
        articles_table$abstractDensity[i] = absDensity
      }
    }
    if ( ! is.null(tdm_paragraph[i,words]) ){
      if (tdm_paragraph[i,words] > 0 ){
        paragraphCount = paragraphCount + tdm_paragraph[i,words]
        pDensity = paragraphCount / articles_table$paragraphTotal[i]
        articles_table$paragraphDensity[i] = pDensity
      }
    }
  }
  abstractCount = 0 
  paragraphCount = 0
}

#lop thru df agn then if smaller than median, 0 else 1
medianWithoutNA<-function(x) {
  median(x[which(!is.na(x))])
}

medians = apply(articles_table[,7:8], 2, medianWithoutNA)

abs_medianDensity = 0.15 
p_medianDensity = 0.1 

articles_table$Findability <- NA

for (i in 1:length(articles_table$Title)){
  if(is.na(articles_table$paragraphDensity[i])){
    articles_table$Findability[i] = 0
  }
  else {
    if(is.na(articles_table$abstractDensity[i])){
      if(articles_table$paragraphDensity[i] >= p_medianDensity){
        articles_table$Findability[i] = 1
      }
      else{
        articles_table$Findability[i] = 0
      }
    }
    else{
      
      if(articles_table$abstractDensity[i] >= abs_medianDensity && articles_table$paragraphDensity[i] >= p_medianDensity){
        articles_table$Findability[i] = 1
      }
      else{
        articles_table$Findability[i] = 0
      }
    }
  }
}

#Replace NaN with 0 in articles_table
articles_table[is.na(articles_table)] <- 0


#split data
train.row <- sample(1:nrow(articles_table), 0.7*nrow(articles_table))
train_data <- articles_table[train.row,]
test_data <- articles_table[-train.row,]

#########################Modelling#####################

train_data$Findability <- as.factor(train_data$Findability)
test_data$Findability <- as.factor(test_data$Findability)


#Random Forest 
rf.fit <- randomForest(Findability ~ Abstract+Paragraphs+Category, train_data, importance = TRUE, ntree = 689)
rf.pred <- predict(rf.fit, test_data)
rf.cfm <- table("actual" = test_data$Findability, "predicted" = rf.pred)
rf.acc <- round(mean(rf.pred == test_data$Findability)*100, digits = 2)
cat("Random Forest model accuracy is: ", rf.acc , "%") 


saveRDS(rf.fit , "./final_model.rds")
