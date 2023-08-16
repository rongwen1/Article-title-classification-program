library(shiny)
library(plotly)
library(tidyr)
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
library(tools)
library(datasets)
library(DT)
library(wordcloud)
library(RColorBrewer)
library(wordcloud2)

clean <- function(x){
  
  x <-removeNumbers(x)
  
  x <-removePunctuation(x)
  
  x <-tolower(x)
  
  x <-removeWords(x,stopwords('en'))
  
  x <-stripWhitespace(x)
  
  x <-stemDocument(x)
  
  return(x) }




ui <- fluidPage(
  
  titlePanel("Classification Model"),
  
  sidebarLayout(
    
    sidebarPanel(
      width = 3,
      style = paste0("height: 90vh; overflow-y: auto;"),
      
      fileInput("file1", "Upload XML File ONLY",
                multiple = FALSE,
                accept = c(".xml")),
      
      selectInput("placement", "Placement:",
                  c("Paragraph", "Abstract")),
      
      actionButton(inputId='ab1', label="Convert PDF to XML", 
                   icon = icon("th"), 
                   onclick ="window.open('https://products.aspose.app/pdf/conversion/pdf-to-xml', '_blank')"),
      
      actionButton(inputId='ab2', label="Predict")
    ),
    mainPanel(
      
      span(strong(textOutput("text")),style= "font-family: 'Times', serif;
    font-weight: 500; font-size: 40px;"),
      
      fluidRow(
        splitLayout(cellWidths = c("50%", "50%"), plotOutput("plot1"),wordcloud2Output("plot2")
        ),
        HTML('<hr style="color: purple;">'),
        dataTableOutput("table2"))
      
    )
  )
)

server <- function(input, output,session) {
  
  
  observeEvent(input$file1, {
    
    input_table = data.frame(matrix(ncol = 4, nrow = 0))
    
    file <- input$file1$datapath
    input_article <- read_xml(file,'UTF_8')
    
    title <- xml_text(xml_find_all(input_article, xpath = "//title-group"))
    abstract <- xml_text(xml_find_all(input_article, xpath = "//abstract"))
    body <- xml_text(xml_find_all(input_article, xpath = "//body"))
    category <- xml_text(xml_find_all(input_article, xpath = "//article-categories"))
    
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
    
    if(abstract == ""){
      
      updateSelectInput(session, "placement", "Placement: (Abstract is empty) ", c("Paragraph"))
      
    }
    
    
  })
  
  
  
  input_file <- eventReactive(input$ab2,{
    input_table = data.frame(matrix(ncol = 4, nrow = 0))
    
    file <- input$file1$datapath
    input_article <- read_xml(file,'UTF_8')
    
    title <- xml_text(xml_find_all(input_article, xpath = "//title-group"))
    abstract <- xml_text(xml_find_all(input_article, xpath = "//abstract"))
    body <- xml_text(xml_find_all(input_article, xpath = "//body"))
    category <- xml_text(xml_find_all(input_article, xpath = "//article-categories"))
    
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
    
    
    input_table <- rbind(input_table, c(title,abstract,body,category))
    
    
    
    colnames(input_table) = c("Title","Abstract","Paragraphs","Category")
    
    
    #################################Create term document matrix########################################################
    
    #Abstract TDM
    myCorpus <- Corpus(VectorSource(input_table$Abstract))
    
    tdm_abstract <- TermDocumentMatrix(myCorpus)
    
    tdm_abstract = as.data.frame(as.matrix(tdm_abstract))
    
    tdm_abstract = t(tdm_abstract)
    
    tdm_abstract = as.data.frame(tdm_abstract, stringsAsFactors = FALSE)
    
    
    #Paragraph TDM
    paraCorpus <- Corpus(VectorSource(input_table$Paragraphs))
    
    tdm_paragraph <- TermDocumentMatrix(paraCorpus)
    
    tdm_paragraph <- removeSparseTerms(tdm_paragraph, 0.7)
    
    tdm_paragraph = as.data.frame(as.matrix(tdm_paragraph))
    
    tdm_paragraph = t(tdm_paragraph)
    
    tdm_paragraph = as.data.frame(tdm_paragraph, stringsAsFactors = FALSE)
    
    
    input_table$abstractTotal <- rowSums(as.matrix(tdm_abstract))
    input_table$paragraphTotal <- rowSums(as.matrix(tdm_paragraph))
    input_table$abstractDensity <- NA
    input_table$paragraphDensity <- NA
    
    abstractCount = 0 
    paragraphCount = 0
    
    for (i in 1:length(input_table$Title)){
      word_list = as.list(scan(text=input_table$Title[i], what="[[:space:]]"))
      for ( words in word_list) {
        if ( abstractCount > 0 ){
          if (tdm_abstract[i,words] > 0 ){
            abstractCount = abstractCount + tdm_abstract[i,words]
            absDensity = abstractCount / input_table$abstractTotal[i]
            input_table$abstractDensity[i] = absDensity
          }
        }
        if ( ! is.null(tdm_paragraph[i,words]) ){
          if (tdm_paragraph[i,words] > 0 ){
            paragraphCount = paragraphCount + tdm_paragraph[i,words]
            pDensity = paragraphCount / input_table$paragraphTotal[i]
            input_table$paragraphDensity[i] = pDensity
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
    
    medians = apply(input_table[,7:8], 2, medianWithoutNA)
    
    abs_medianDensity = 0.15 
    p_medianDensity = 0.1 
    
    input_table$Findability <- NA
    
    for (i in 1:length(input_table$Title)){
      if(is.na(input_table$paragraphDensity[i])){
        input_table$Findability[i] = 0
      }
      else {
        if(is.na(input_table$abstractDensity[i])){
          if(input_table$paragraphDensity[i] >= p_medianDensity){
            input_table$Findability[i] = 1
          }
          else{
            input_table$Findability[i] = 0
          }
        }
        else{
          
          if(input_table$abstractDensity[i] >= abs_medianDensity && input_table$paragraphDensity[i] >= p_medianDensity){
            input_table$Findability[i] = 1
          }
          else{
            input_table$Findability[i] = 0
          }
        }
      }
    }
    
    #Replace NaN with 0 in input_table
    input_table[is.na(input_table)] <- 0
    input_table
  })
  
  classification_model <- eventReactive(input$ab2,{
    
    input_table <- input_file()
    
    model <- readRDS("./final_model.rds")
    pred <- predict(model, input_table)
    rf.cfm <- table("actual" = input_table$Findability, "predicted" = pred)
    rf.acc <- round(mean(pred == input_table$Findability)*100, digits = 2)
    if (pred[1] == 1){
      res = "The uploaded article has a great findability based on the Title"
    }
    else{
      res = "The uploaded article has a poor findability based on the Title"
    }
    
    res
  })
  
  
  
  density_plot <- eventReactive(input$ab2,{
    
    user_input_article = input_file()
    
    #Create list for each placement
    input_title = (scan(text=user_input_article$Title, what="[[:space:]]"))
    input_abstract = as.list(scan(text=user_input_article$Abstract, what="[[:space:]]"))
    input_paragraph = as.list(scan(text=user_input_article$Paragraph, what="[[:space:]]"))
    
    
    #Create new data frame to store Title words statistics
    input_df = as.data.frame((scan(text=user_input_article$Title, what="[[:space:]]")))
    names(input_df)[1] <- 'Title'
    input_df["Abstract"] = 0
    input_df["Paragraph"] = 0
    
    
    
    #Length for each placement
    abstract_total = length(input_abstract)
    paragraph_total = length(input_paragraph)
    
    #Calculate 
    for (i in 1:length(input_title)){
      
      #Abstract
      abstract_count = 0
      if (length(input_abstract) > 0){
        for (j in 1:length(input_abstract)){
          if (input_title[i] == input_abstract[j]){
            abstract_count = abstract_count + 1
          }
        }
      }
      input_df[i, 2] = abstract_count
      
      
      #Paragraph
      paragraph_count = 0
      if (length(input_paragraph) > 0){
        for (j in 1:length(input_paragraph)){
          if (input_title[i] == input_paragraph[j]){
            paragraph_count = paragraph_count + 1
          }
        }
      }
      input_df[i, 3] = paragraph_count
      
      
    }
    
    
    #Process data for plotting
    plotting = t(input_df)
    colnames(plotting) <- plotting[1,]
    plotting <- plotting[-1,] 
    
    #####Bar chart for each placements#####
    if (input$placement == "Abstract"){
      plot_abstract = t(plotting["Abstract",][order(plotting["Abstract",])])  #Filter placement, then sort by ascending
      #Pick the top 10 occurrence
      if (length(plot_abstract) > 10){
        plot_abstract = t(plot_abstract[,(ncol(plot_abstract)-9):ncol(plot_abstract)])
      }
      plot = barplot(as.matrix(plot_abstract),
                     col = "blue",
                     main = "Word counts of Title in abstract",
                     xlab = "Words",
                     ylab = "Occurrence")
      
    }
    else {
      plot_paragraph = t(plotting["Paragraph",][order(plotting["Paragraph",])])
      if (length(plot_paragraph) > 10){
        plot_paragraph = t(plot_paragraph[,(ncol(plot_paragraph)-9):ncol(plot_paragraph)])
      }
      plot= barplot(as.matrix(plot_paragraph),
                    col = "lightblue",
                    main = "Word counts of Title in Paragraph",
                    xlab = "Words",
                    ylab = "Occurrence")
    }
    
    plot
    
  })
  
  
  related_keywords <- eventReactive(input$ab2,{
    
    file <- input$file1$datapath
    input_article <- read_xml(file,'UTF_8')
    
    title <- xml_text(xml_find_all(input_article, xpath = "//title-group"))
    abstract <- xml_text(xml_find_all(input_article, xpath = "//abstract"))
    body <- xml_text(xml_find_all(input_article, xpath = "//body"))
    category <- xml_text(xml_find_all(input_article, xpath = "//article-categories"))
    
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
    
    complete = paste(abstract, body)
    
    myCorpus <- Corpus(VectorSource(complete))
    
    tdm <- TermDocumentMatrix(myCorpus)
    
    res = as.data.frame(findMostFreqTerms(tdm,50L))
    
    res
    
  })
  
  word_cloud <- eventReactive(input$ab2,{
    
    
    
    input_table = data.frame(matrix(ncol = 4, nrow = 0))
    
    file <- input$file1$datapath
    input_article <- read_xml(file,'UTF_8')
    
    title <- xml_text(xml_find_all(input_article, xpath = "//title-group"))
    abstract <- xml_text(xml_find_all(input_article, xpath = "//abstract"))
    body <- xml_text(xml_find_all(input_article, xpath = "//body"))
    category <- xml_text(xml_find_all(input_article, xpath = "//article-categories"))
    
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
    
    input_table <- rbind(input_table, c(title,abstract,body,category))
    
    
    
    colnames(input_table) = c("Title","Abstract","Paragraphs","Category")
    
    
    #################################Create term document matrix########################################################
    
    #Abstract TDM
    if(input$placement== 'Abstract'){
      myCorpus <- Corpus(VectorSource(input_table$Abstract))
      
      tdm_abstract <- TermDocumentMatrix(myCorpus)
      
      tdm_abstract_matrix = as.matrix(tdm_abstract)
      
      words <- sort(rowSums(tdm_abstract_matrix),decreasing=TRUE) 
      
      df_abstract <- data.frame(word = names(words),freq=words)
      
      wc = wordcloud2(df_abstract)
      
    }
    else{
      #Paragraph TDM
      paraCorpus <- Corpus(VectorSource(input_table$Paragraphs))
      
      tdm_paragraph <- TermDocumentMatrix(paraCorpus)
      
      tdm_paragraph <- removeSparseTerms(tdm_paragraph, 0.7)
      
      tdm_paragraph_matrix = as.matrix(tdm_paragraph)
      
      words <- sort(rowSums(tdm_paragraph_matrix),decreasing=TRUE) 
      
      df_paragraph <- data.frame(word = names(words),freq=words)
      
      wc = wordcloud2(df_paragraph)
      
    }
    
    wc
  })
  
  observeEvent(input$ab2,{
    
    output$table2 <- renderDataTable({
      
      related_keywords()
      
    })
  })
  
  observeEvent(input$ab2,{
    
    output$text <- renderText({
      
      classification_model()
      
    })
  })
  
  observeEvent(input$ab2,{
    
    output$plot1 <- renderPlot({
      
      density_plot()
      
    })
  })
  
  observeEvent(input$ab2,{
    
    output$plot2 <- renderWordcloud2({
      
      word_cloud()
      
    })
  })
  
}


shinyApp(ui, server)