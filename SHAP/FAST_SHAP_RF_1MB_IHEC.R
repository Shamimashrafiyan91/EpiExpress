# print("Fast for Random Forest with paralization 15 Aug 2025")
# library(randomForest)
# library(randomForestExplainer)
# library(fastshap)
# library(readr)
# 
# library(plyr)
# library(boot)
# library(tfruns)
# library(MLmetrics)
# library(parallel)
# library(snow)
# # install.packages("")
# #install.packages("tidyverse")
# library(tidyverse)
# library(reticulate)
# library(lime)
# library(foreach)
# library(doParallel)
# 
# normalize_min_max <- function(data) {
#   if(max(data) == min(data)){
#     return(0)
#   }
# 
#   return ((data - min(data)) / (max(data) - min(data)))
# }
# 
# 
# 
# # gene_list = c("ENSG00000140545", "ENSG00000135363", "ENSG00000123405")
# lst =list()
# 
# # file_path = ("/projects/apog/work/models/1MB/RF/RF_last/kept_genes.txt")
# # file_contents <- read.table(file_path, header = FALSE, col.names = c("Index", "Gene"))$Gene
# # file_contents = file_contents[1:2]
# # #dont forget to upload it
# # myCon = file_contents
# # gene_names_to_process <- myCon
# 
# file_path = ("/projects/apog/work/models/1MB/genes_CRISPR.txt")
# myCon <- readLines(file_path)
# print(length(myCon))
# 
#  gene_names_to_process <- myCon
# # process_gene_name <- function(gene_name){
# for(gene_name in myCon){
#   lst <- list()
#   tryCatch({
# 
#     print(gene_name)
# 
#     my_model <- readRDS(paste0("/projects/apog/work/models/1MB/RF/RF_last/", gene_name,".RDS"))
# 
#     my_data = read.table(paste0("/projects/apog/work/input/IHEC_Activity_1MB_hg38/", gene_name, ".txt.gz"), header = TRUE, sep = "\t" )
# 
#     df = my_data
#     df <- df[order(df$Sample),] #ordering the sample names
#     rownames(df) = NULL
#     df = df[, colSums(df != 0) > 0] #if all values of a column is 0 Ill remove that column
#     df_samples = df[,1] #GET THE SAMPLE name, first col
#     df2 = df[, 2:length(df)]
#     df2 = log2(df2 + 1) #whole data after log2,
#     df3 = cbind(df_samples, df2) #df3 is my dataset after log2 + 1
# 
#     train_samples <- "/projects/apog/work/CNN_bin/miscellaneous/partition0_train.csv"
# 
# 
#     train_samples2 <- read.csv(train_samples)
#     train_samples3 <- train_samples2[,1]
#     train_data <- df3[df3$df_samples %in% train_samples3, ]
#     sample_final <- train_data[,1]
# 
#     #rename the train and test data
#     train_data = train_data[, 2:length(train_data)] #whole
#     train_traget_just_log_no_norm = train_data[,ncol(train_data)] #target original
#     train_data2 = train_data[,1: (ncol(train_data)) - 1] #whole without target
# 
#     train_data_normalized <- as.data.frame(apply(train_data2, 2, normalize_min_max))
#     rownames(train_data_normalized) <- NULL
#     training <- train_data_normalized #for train data without sample names
#     print("#########################################################################################")
#     rownames(train_data_normalized) <- sample_final
#     train_K562 <- train_data_normalized[rownames(train_data_normalized) == "IHECRE00001887",]
#     rownames(train_K562) <- NULL #instance without sample name
# 
#     your_data <- training  # Your input data as a data frame
#     # my_model <- randomForest(formula, data = your_data)  # Training your randomForest model
#     my_instance = train_K562
# 
# 
# 
#     RF_predict_wrapper <- function(model, newdata) {
#       predictions <- predict(model, newdata)
#       return(predictions)
#     }
# 
#     set.seed(200)  # for reproducibility
#     print("starting the SHAP process")
#    SHAP_result2 <- fastshap::explain(my_model, X = training, pred_wrapper = RF_predict_wrapper, newdata = my_instance,
#                                        nsim =  3 * (ncol(training)), adjust = FALSE)
# 
#     print("finishing shap process")
# 
#     lst[[gene_name]] = SHAP_result2
#     name1 = paste0("/projects/apog/work/models/1MB/SHAP/RF_review/SHAP_RF_",gene_name,".RDS")
#     saveRDS(lst,name1)
# 
#     #
# 
# 
# 
# 
#   },  error=function(e){
#     print(paste("Error occurred at gene", gene_name)); message(e)
# 
# 
#   })
# 
# 
# 
# 
#   # return(lst)
# 
# }
# 
# 
# 
# 
# 
# 
# print("end of the Shap code for RF")





print("Fast for Random Forest with parallelization 15 Aug 2025")
library(randomForest)
library(randomForestExplainer)
library(fastshap)
library(readr)
library(plyr) 
library(boot)
library(tfruns)
library(MLmetrics)
library(parallel)
library(snow)
library(tidyverse)
library(reticulate)
library(lime)
library(foreach)
library(doParallel)

# Set up logging
log_file <- "/projects/apog/work/models/1MB/SHAP/RF_review/SHAP_processing.log"
sink(log_file, append = TRUE, split = TRUE)

# Normalization function
normalize_min_max <- function(data) {
  if(max(data) == min(data)) {
    return(0)
  }
  return ((data - min(data)) / (max(data) - min(data)))
}

# Configuration
gene_file_path <- "/projects/apog/work/models/1MB/genes_CRISPR.txt"
model_dir <- "/projects/apog/work/models/1MB/RF/RF_last/"
data_dir <- "/projects/apog/work/input/IHEC_Activity_1MB_hg38/"
output_dir <- "/projects/apog/work/models/1MB/SHAP/RF_review/"
train_samples_path <- "/projects/apog/work/CNN_bin/miscellaneous/partition0_train.csv"

# Create output directory if it doesn't exist
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Read gene list
gene_names_to_process <- readLines(gene_file_path)
print(paste("Total genes to process:", length(gene_names_to_process)))

# Random Forest prediction wrapper
RF_predict_wrapper <- function(model, newdata) {
  predictions <- predict(model, newdata)
  return(predictions)
}

# Function to process a single gene
process_gene <- function(gene_name) {
  result <- list()
  tryCatch({
    # Log start
    print(paste("Starting processing for gene:", gene_name))
    start_time <- Sys.time()
    
    # Load model
    model_path <- paste0(model_dir, gene_name, ".RDS")
    if (!file.exists(model_path)) {
      stop(paste("Model file not found for gene:", gene_name))
    }
    my_model <- readRDS(model_path)
    
    # Load data
    data_path <- paste0(data_dir, gene_name, ".txt.gz")
    if (!file.exists(data_path)) {
      stop(paste("Data file not found for gene:", gene_name))
    }
    my_data <- read.table(data_path, header = TRUE, sep = "\t")
    
    # Preprocess data
    df <- my_data[order(my_data$Sample),]
    rownames(df) <- NULL
    df <- df[, colSums(df != 0) > 0]
    
    df_samples <- df[,1]
    df2 <- log2(df[, 2:ncol(df)] + 1)
    df3 <- cbind(df_samples, df2)
    
    # Get training samples
    train_samples <- read.csv(train_samples_path)[,1]
    train_data <- df3[df3$df_samples %in% train_samples, ]
    sample_final <- train_data[,1]
    
    # Prepare training data
    train_data <- train_data[, 2:ncol(train_data)]
    train_data2 <- train_data[, 1:(ncol(train_data) - 1)]
    
    train_data_normalized <- as.data.frame(apply(train_data2, 2, normalize_min_max))
    rownames(train_data_normalized) <- NULL
    training <- train_data_normalized
    
    # Prepare K562 instance
    rownames(train_data_normalized) <- sample_final
    train_K562 <- train_data_normalized[rownames(train_data_normalized) == "IHECRE00001887",]
    rownames(train_K562) <- NULL
    
    # Compute SHAP values
    print(paste("Starting SHAP computation for gene:", gene_name))
    SHAP_result <- fastshap::explain(
      my_model, 
      X = training, 
      pred_wrapper = RF_predict_wrapper, 
      newdata = train_K562,
      nsim = 3 * ncol(training), 
      adjust = FALSE
    )
    
    # Save results
    result[[gene_name]] <- SHAP_result
    output_path <- paste0(output_dir, "SHAP_RF_", gene_name, ".RDS")
    saveRDS(result, output_path)
    
    # Log completion
    end_time <- Sys.time()
    duration <- difftime(end_time, start_time, units = "mins")
    print(paste("Successfully processed gene:", gene_name, "in", round(duration, 2), "minutes"))
    
    return(list(gene = gene_name, status = "success", time = duration))
    
  }, error = function(e) {
    # Log error
    error_msg <- paste("Error processing gene", gene_name, ":", conditionMessage(e))
    print(error_msg)
    return(list(gene = gene_name, status = "error", error = error_msg))
  })
}

# Set up parallel processing
num_cores <- detectCores() - 22 # Leave 2 cores free
cl <- makeCluster(num_cores)
registerDoParallel(cl)

# Process genes in parallel with progress tracking
print(paste("Starting parallel processing with", num_cores, "cores"))
start_total <- Sys.time()

results <- foreach(gene = gene_names_to_process, 
                   .packages = c("randomForest", "fastshap", "readr"),
                   .combine = 'rbind',
                   .errorhandling = 'pass') %dopar% {
                     process_gene(gene)
                   }

# Clean up parallel processing
stopCluster(cl)

# Generate summary report
success_count <- sum(sapply(results, function(x) x$status == "success"))
error_count <- length(gene_names_to_process) - success_count

total_time <- difftime(Sys.time(), start_total, units = "hours")

print("################################################################")
print(paste("Processing completed for", length(gene_names_to_process), "genes"))
print(paste("Successfully processed:", success_count))
print(paste("Failed:", error_count))
print(paste("Total processing time:", round(total_time, 2), "hours"))
print("################################################################")

# Save processing summary
summary_df <- data.frame(
  Gene = sapply(results, function(x) x$gene),
  Status = sapply(results, function(x) x$status),
  Time = sapply(results, function(x) ifelse(!is.null(x$time), round(x$time, 2), NA)),
  Error = sapply(results, function(x) ifelse(!is.null(x$error), x$error, NA))
)

write.csv(summary_df, paste0(output_dir, "processing_summary.csv"), row.names = FALSE)

# Close log
sink()
print("End of the SHAP code for RF")