# Advanced R Risk Analysis
# Statistical analysis and visualization for financial risk assessment
# 
# Technologies: R, Shiny, ggplot2, dplyr, caret, randomForest, xgboost
# Author: Nithin Yanna
# Date: 2025

# Load required libraries
library(shiny)
library(ggplot2)
library(dplyr)
library(caret)
library(randomForest)
library(xgboost)
library(corrplot)
library(VIM)
library(ROCR)
library(pROC)
library(plotly)
library(DT)
library(shinydashboard)
library(shinyWidgets)
library(leaflet)
library(htmlwidgets)
library(jsonlite)
library(httr)
library(lubridate)
library(stringr)
library(tidyr)
library(reshape2)
library(gridExtra)
library(RColorBrewer)
library(scales)
library(forecast)
library(tseries)
library(cluster)
library(factoextra)
library(psych)
library(car)
library(MASS)
library(glmnet)
library(e1071)
library(nnet)
library(neuralnet)
library(deepnet)
library(h2o)
library(mlr)
library(parallelMap)
library(parallel)

# Global configuration
options(scipen = 999)
set.seed(42)

# Data loading and preprocessing functions
load_risk_data <- function(file_path) {
  # Load data from various sources
  if (str_detect(file_path, "\\.csv$")) {
    data <- read.csv(file_path, stringsAsFactors = FALSE)
  } else if (str_detect(file_path, "\\.json$")) {
    data <- fromJSON(file_path)
  } else if (str_detect(file_path, "\\.xlsx$")) {
    data <- readxl::read_excel(file_path)
  } else {
    stop("Unsupported file format")
  }
  
  # Convert to data frame if needed
  if (!is.data.frame(data)) {
    data <- as.data.frame(data)
  }
  
  return(data)
}

# Data quality assessment
assess_data_quality <- function(data) {
  quality_report <- list()
  
  # Basic statistics
  quality_report$total_rows <- nrow(data)
  quality_report$total_columns <- ncol(data)
  quality_report$missing_values <- sum(is.na(data))
  quality_report$missing_percentage <- (quality_report$missing_values / (nrow(data) * ncol(data))) * 100
  
  # Column-wise missing values
  quality_report$missing_by_column <- colSums(is.na(data))
  quality_report$missing_percentage_by_column <- (quality_report$missing_by_column / nrow(data)) * 100
  
  # Data types
  quality_report$data_types <- sapply(data, class)
  
  # Duplicate rows
  quality_report$duplicate_rows <- sum(duplicated(data))
  quality_report$duplicate_percentage <- (quality_report$duplicate_rows / nrow(data)) * 100
  
  # Outliers detection (using IQR method)
  numeric_columns <- sapply(data, is.numeric)
  if (sum(numeric_columns) > 0) {
    outliers_count <- 0
    for (col in names(data)[numeric_columns]) {
      Q1 <- quantile(data[[col]], 0.25, na.rm = TRUE)
      Q3 <- quantile(data[[col]], 0.75, na.rm = TRUE)
      IQR <- Q3 - Q1
      outliers <- data[[col]] < (Q1 - 1.5 * IQR) | data[[col]] > (Q3 + 1.5 * IQR)
      outliers_count <- outliers_count + sum(outliers, na.rm = TRUE)
    }
    quality_report$outliers_count <- outliers_count
  }
  
  return(quality_report)
}

# Advanced feature engineering
engineer_features <- function(data) {
  # Create a copy of the data
  features_data <- data
  
  # Time-based features
  if ("timestamp" %in% names(data)) {
    features_data$timestamp <- as.POSIXct(features_data$timestamp)
    features_data$hour <- hour(features_data$timestamp)
    features_data$day_of_week <- wday(features_data$timestamp)
    features_data$month <- month(features_data$timestamp)
    features_data$is_weekend <- ifelse(features_data$day_of_week %in% c(1, 7), 1, 0)
    features_data$is_night <- ifelse(features_data$hour >= 22 | features_data$hour <= 6, 1, 0)
  }
  
  # Amount-based features
  if ("amount" %in% names(data)) {
    features_data$amount_log <- log(features_data$amount + 1)
    features_data$amount_sqrt <- sqrt(features_data$amount)
    features_data$amount_squared <- features_data$amount^2
    features_data$amount_cubed <- features_data$amount^3
  }
  
  # Customer-based features
  if ("customer_id" %in% names(data)) {
    # Customer transaction count
    customer_counts <- table(features_data$customer_id)
    features_data$customer_transaction_count <- customer_counts[features_data$customer_id]
    
    # Customer average amount
    customer_avg_amount <- aggregate(amount ~ customer_id, data = features_data, FUN = mean)
    features_data$customer_avg_amount <- customer_avg_amount$amount[match(features_data$customer_id, customer_avg_amount$customer_id)]
    
    # Customer risk history
    if ("risk_score" %in% names(data)) {
      customer_risk_history <- aggregate(risk_score ~ customer_id, data = features_data, FUN = mean)
      features_data$customer_risk_history <- customer_risk_history$risk_score[match(features_data$customer_id, customer_risk_history$customer_id)]
    }
  }
  
  # Location-based features
  if ("location" %in% names(data)) {
    # Location transaction count
    location_counts <- table(features_data$location)
    features_data$location_transaction_count <- location_counts[features_data$location]
    
    # Location average amount
    location_avg_amount <- aggregate(amount ~ location, data = features_data, FUN = mean)
    features_data$location_avg_amount <- location_avg_amount$amount[match(features_data$location, location_avg_amount$location)]
  }
  
  # Interaction features
  if ("amount" %in% names(data) && "customer_age" %in% names(data)) {
    features_data$amount_age_ratio <- features_data$amount / features_data$customer_age
  }
  
  if ("amount" %in% names(data) && "customer_transaction_count" %in% names(data)) {
    features_data$amount_frequency_ratio <- features_data$amount / features_data$customer_transaction_count
  }
  
  return(features_data)
}

# Advanced machine learning models
train_ensemble_model <- function(data, target_column = "risk_score") {
  # Prepare data
  features <- data[, !names(data) %in% c(target_column, "transaction_id", "customer_id", "timestamp")]
  
  # Remove non-numeric columns for now
  numeric_features <- features[, sapply(features, is.numeric)]
  
  # Handle missing values
  numeric_features[is.na(numeric_features)] <- 0
  
  # Create target variable
  if (target_column %in% names(data)) {
    target <- data[[target_column]]
  } else {
    # Create synthetic target for demonstration
    target <- rnorm(nrow(data), 0.5, 0.2)
    target <- pmax(0, pmin(1, target))  # Clamp between 0 and 1
  }
  
  # Split data
  train_index <- createDataPartition(target, p = 0.8, list = FALSE)
  train_data <- numeric_features[train_index, ]
  test_data <- numeric_features[-train_index, ]
  train_target <- target[train_index]
  test_target <- target[-train_index]
  
  # Train multiple models
  models <- list()
  
  # Random Forest
  rf_model <- randomForest(x = train_data, y = train_target, ntree = 100, mtry = sqrt(ncol(train_data)))
  models$random_forest <- rf_model
  
  # XGBoost
  xgb_model <- xgboost(
    data = as.matrix(train_data),
    label = train_target,
    nrounds = 100,
    objective = "reg:squarederror",
    verbose = 0
  )
  models$xgboost <- xgb_model
  
  # Neural Network
  nn_model <- nnet(x = train_data, y = train_target, size = 10, linout = TRUE, trace = FALSE)
  models$neural_network <- nn_model
  
  # Support Vector Machine
  svm_model <- svm(x = train_data, y = train_target, kernel = "radial")
  models$svm <- svm_model
  
  # Linear Regression
  lm_model <- lm(train_target ~ ., data = train_data)
  models$linear_regression <- lm_model
  
  # Make predictions
  predictions <- list()
  predictions$random_forest <- predict(rf_model, test_data)
  predictions$xgboost <- predict(xgb_model, as.matrix(test_data))
  predictions$neural_network <- predict(nn_model, test_data)
  predictions$svm <- predict(svm_model, test_data)
  predictions$linear_regression <- predict(lm_model, test_data)
  
  # Calculate ensemble prediction
  ensemble_pred <- rowMeans(do.call(cbind, predictions))
  
  # Calculate performance metrics
  performance <- list()
  for (model_name in names(predictions)) {
    rmse <- sqrt(mean((test_target - predictions[[model_name]])^2))
    mae <- mean(abs(test_target - predictions[[model_name]]))
    r_squared <- cor(test_target, predictions[[model_name]])^2
    
    performance[[model_name]] <- list(
      rmse = rmse,
      mae = mae,
      r_squared = r_squared
    )
  }
  
  # Ensemble performance
  ensemble_rmse <- sqrt(mean((test_target - ensemble_pred)^2))
  ensemble_mae <- mean(abs(test_target - ensemble_pred))
  ensemble_r_squared <- cor(test_target, ensemble_pred)^2
  
  performance$ensemble <- list(
    rmse = ensemble_rmse,
    mae = ensemble_mae,
    r_squared = ensemble_r_squared
  )
  
  return(list(
    models = models,
    predictions = predictions,
    performance = performance,
    ensemble_prediction = ensemble_pred,
    test_target = test_target
  ))
}

# Advanced visualization functions
create_risk_distribution_plot <- function(data, risk_column = "risk_score") {
  if (!risk_column %in% names(data)) {
    # Create synthetic risk scores for demonstration
    data$risk_score <- rnorm(nrow(data), 0.5, 0.2)
    data$risk_score <- pmax(0, pmin(1, data$risk_score))
  }
  
  p <- ggplot(data, aes_string(x = risk_column)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = "steelblue", alpha = 0.7) +
    geom_density(color = "red", size = 1) +
    geom_vline(aes(xintercept = mean(get(risk_column), na.rm = TRUE)), 
               color = "green", linetype = "dashed", size = 1) +
    geom_vline(aes(xintercept = median(get(risk_column), na.rm = TRUE)), 
               color = "orange", linetype = "dashed", size = 1) +
    labs(
      title = "Risk Score Distribution",
      x = "Risk Score",
      y = "Density"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10)
    )
  
  return(p)
}

create_correlation_heatmap <- function(data, numeric_columns = NULL) {
  if (is.null(numeric_columns)) {
    numeric_columns <- names(data)[sapply(data, is.numeric)]
  }
  
  if (length(numeric_columns) < 2) {
    return(NULL)
  }
  
  correlation_matrix <- cor(data[numeric_columns], use = "complete.obs")
  
  # Create heatmap
  melted_cor <- melt(correlation_matrix)
  
  p <- ggplot(melted_cor, aes(Var1, Var2, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(
      low = "blue", 
      mid = "white", 
      high = "red",
      midpoint = 0,
      limit = c(-1, 1),
      space = "Lab",
      name = "Correlation"
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      axis.title.x = element_blank(),
      axis.title.y = element_blank()
    ) +
    coord_fixed()
  
  return(p)
}

create_time_series_plot <- function(data, time_column = "timestamp", value_column = "risk_score") {
  if (!time_column %in% names(data)) {
    # Create synthetic timestamp
    data$timestamp <- seq(from = as.POSIXct("2024-01-01"), 
                        to = as.POSIXct("2024-12-31"), 
                        length.out = nrow(data))
  }
  
  if (!value_column %in% names(data)) {
    # Create synthetic risk scores
    data$risk_score <- rnorm(nrow(data), 0.5, 0.2)
    data$risk_score <- pmax(0, pmin(1, data$risk_score))
  }
  
  # Convert to time series
  data$timestamp <- as.POSIXct(data$timestamp)
  data <- data[order(data$timestamp), ]
  
  p <- ggplot(data, aes_string(x = time_column, y = value_column)) +
    geom_line(color = "steelblue", size = 1) +
    geom_smooth(method = "loess", color = "red", se = TRUE) +
    labs(
      title = "Risk Score Time Series",
      x = "Time",
      y = "Risk Score"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 16, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10)
    )
  
  return(p)
}

# Shiny dashboard application
ui <- dashboardPage(
  dashboardHeader(title = "Advanced Risk Analysis Dashboard"),
  
  dashboardSidebar(
    sidebarMenu(
      menuItem("Data Overview", tabName = "overview", icon = icon("dashboard")),
      menuItem("Risk Analysis", tabName = "analysis", icon = icon("chart-line")),
      menuItem("Machine Learning", tabName = "ml", icon = icon("brain")),
      menuItem("Visualizations", tabName = "viz", icon = icon("chart-bar")),
      menuItem("Model Performance", tabName = "performance", icon = icon("tachometer-alt")),
      menuItem("Data Quality", tabName = "quality", icon = icon("check-circle"))
    )
  ),
  
  dashboardBody(
    tabItems(
      # Data Overview Tab
      tabItem(
        tabName = "overview",
        fluidRow(
          box(
            title = "Data Summary", status = "primary", solidHeader = TRUE,
            width = 12,
            verbatimTextOutput("data_summary")
          )
        ),
        fluidRow(
          box(
            title = "Data Quality Report", status = "info", solidHeader = TRUE,
            width = 12,
            verbatimTextOutput("quality_report")
          )
        )
      ),
      
      # Risk Analysis Tab
      tabItem(
        tabName = "analysis",
        fluidRow(
          box(
            title = "Risk Score Distribution", status = "primary", solidHeader = TRUE,
            width = 6,
            plotOutput("risk_distribution")
          ),
          box(
            title = "Risk Statistics", status = "success", solidHeader = TRUE,
            width = 6,
            verbatimTextOutput("risk_statistics")
          )
        ),
        fluidRow(
          box(
            title = "Correlation Heatmap", status = "warning", solidHeader = TRUE,
            width = 12,
            plotOutput("correlation_heatmap")
          )
        )
      ),
      
      # Machine Learning Tab
      tabItem(
        tabName = "ml",
        fluidRow(
          box(
            title = "Model Training", status = "primary", solidHeader = TRUE,
            width = 12,
            actionButton("train_models", "Train Models", class = "btn-primary"),
            verbatimTextOutput("training_status")
          )
        ),
        fluidRow(
          box(
            title = "Model Performance", status = "success", solidHeader = TRUE,
            width = 12,
            verbatimTextOutput("model_performance")
          )
        ),
        fluidRow(
          box(
            title = "Feature Importance", status = "info", solidHeader = TRUE,
            width = 12,
            plotOutput("feature_importance")
          )
        )
      ),
      
      # Visualizations Tab
      tabItem(
        tabName = "viz",
        fluidRow(
          box(
            title = "Time Series Analysis", status = "primary", solidHeader = TRUE,
            width = 12,
            plotOutput("time_series")
          )
        ),
        fluidRow(
          box(
            title = "Risk Level Distribution", status = "success", solidHeader = TRUE,
            width = 6,
            plotOutput("risk_level_distribution")
          ),
          box(
            title = "Amount vs Risk Score", status = "warning", solidHeader = TRUE,
            width = 6,
            plotOutput("amount_risk_scatter")
          )
        )
      ),
      
      # Model Performance Tab
      tabItem(
        tabName = "performance",
        fluidRow(
          box(
            title = "Model Comparison", status = "primary", solidHeader = TRUE,
            width = 12,
            plotOutput("model_comparison")
          )
        ),
        fluidRow(
          box(
            title = "Prediction vs Actual", status = "success", solidHeader = TRUE,
            width = 12,
            plotOutput("prediction_actual")
          )
        )
      ),
      
      # Data Quality Tab
      tabItem(
        tabName = "quality",
        fluidRow(
          box(
            title = "Missing Values Analysis", status = "primary", solidHeader = TRUE,
            width = 12,
            plotOutput("missing_values_plot")
          )
        ),
        fluidRow(
          box(
            title = "Outlier Detection", status = "warning", solidHeader = TRUE,
            width = 12,
            plotOutput("outlier_plot")
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {
  # Global reactive values
  values <- reactiveValues(
    data = NULL,
    quality_report = NULL,
    models = NULL,
    predictions = NULL,
    performance = NULL
  )
  
  # Load sample data on startup
  observe({
    # Create sample data for demonstration
    n <- 1000
    sample_data <- data.frame(
      transaction_id = paste0("TXN", 1:n),
      customer_id = paste0("CUST", sample(1:100, n, replace = TRUE)),
      amount = rlnorm(n, meanlog = 6, sdlog = 1),
      timestamp = seq(from = as.POSIXct("2024-01-01"), 
                     to = as.POSIXct("2024-12-31"), 
                     length.out = n),
      location = sample(c("US", "EU", "ASIA", "LATAM"), n, replace = TRUE),
      customer_age = sample(18:80, n, replace = TRUE),
      payment_method = sample(c("CREDIT", "DEBIT", "CASH", "DIGITAL"), n, replace = TRUE),
      risk_score = rnorm(n, 0.5, 0.2)
    )
    
    # Clamp risk scores between 0 and 1
    sample_data$risk_score <- pmax(0, pmin(1, sample_data$risk_score))
    
    values$data <- sample_data
    values$quality_report <- assess_data_quality(sample_data)
  })
  
  # Data summary output
  output$data_summary <- renderPrint({
    if (!is.null(values$data)) {
      cat("Dataset Summary:\n")
      cat("================\n")
      cat("Rows:", nrow(values$data), "\n")
      cat("Columns:", ncol(values$data), "\n")
      cat("Column names:", paste(names(values$data), collapse = ", "), "\n\n")
      
      cat("Data types:\n")
      print(sapply(values$data, class))
      cat("\nFirst few rows:\n")
      print(head(values$data))
    }
  })
  
  # Quality report output
  output$quality_report <- renderPrint({
    if (!is.null(values$quality_report)) {
      cat("Data Quality Report:\n")
      cat("===================\n")
      cat("Total rows:", values$quality_report$total_rows, "\n")
      cat("Total columns:", values$quality_report$total_columns, "\n")
      cat("Missing values:", values$quality_report$missing_values, "\n")
      cat("Missing percentage:", round(values$quality_report$missing_percentage, 2), "%\n")
      cat("Duplicate rows:", values$quality_report$duplicate_rows, "\n")
      cat("Duplicate percentage:", round(values$quality_report$duplicate_percentage, 2), "%\n")
      
      if (!is.null(values$quality_report$outliers_count)) {
        cat("Outliers detected:", values$quality_report$outliers_count, "\n")
      }
    }
  })
  
  # Risk distribution plot
  output$risk_distribution <- renderPlot({
    if (!is.null(values$data)) {
      create_risk_distribution_plot(values$data)
    }
  })
  
  # Risk statistics
  output$risk_statistics <- renderPrint({
    if (!is.null(values$data) && "risk_score" %in% names(values$data)) {
      cat("Risk Score Statistics:\n")
      cat("=====================\n")
      print(summary(values$data$risk_score))
      cat("\nStandard deviation:", round(sd(values$data$risk_score, na.rm = TRUE), 4), "\n")
      cat("Variance:", round(var(values$data$risk_score, na.rm = TRUE), 4), "\n")
    }
  })
  
  # Correlation heatmap
  output$correlation_heatmap <- renderPlot({
    if (!is.null(values$data)) {
      create_correlation_heatmap(values$data)
    }
  })
  
  # Time series plot
  output$time_series <- renderPlot({
    if (!is.null(values$data)) {
      create_time_series_plot(values$data)
    }
  })
  
  # Model training
  observeEvent(input$train_models, {
    if (!is.null(values$data)) {
      withProgress(message = "Training models...", {
        # Engineer features
        engineered_data <- engineer_features(values$data)
        
        # Train ensemble model
        setProgress(0.5, message = "Training ensemble model...")
        result <- train_ensemble_model(engineered_data)
        
        values$models <- result$models
        values$predictions <- result$predictions
        values$performance <- result$performance
        
        setProgress(1, message = "Training completed!")
      })
    }
  })
  
  # Training status
  output$training_status <- renderPrint({
    if (!is.null(values$models)) {
      cat("Model Training Status: COMPLETED\n")
      cat("Models trained:", length(values$models), "\n")
      cat("Models: Random Forest, XGBoost, Neural Network, SVM, Linear Regression\n")
    } else {
      cat("Click 'Train Models' to start training...\n")
    }
  })
  
  # Model performance
  output$model_performance <- renderPrint({
    if (!is.null(values$performance)) {
      cat("Model Performance Metrics:\n")
      cat("========================\n")
      for (model_name in names(values$performance)) {
        cat("\n", model_name, ":\n")
        cat("  RMSE:", round(values$performance[[model_name]]$rmse, 4), "\n")
        cat("  MAE:", round(values$performance[[model_name]]$mae, 4), "\n")
        cat("  R-squared:", round(values$performance[[model_name]]$r_squared, 4), "\n")
      }
    }
  })
  
  # Feature importance plot
  output$feature_importance <- renderPlot({
    if (!is.null(values$models) && "random_forest" %in% names(values$models)) {
      rf_model <- values$models$random_forest
      importance_df <- data.frame(
        feature = names(rf_model$importance),
        importance = as.numeric(rf_model$importance)
      )
      importance_df <- importance_df[order(importance_df$importance, decreasing = TRUE), ]
      importance_df <- head(importance_df, 10)
      
      ggplot(importance_df, aes(x = reorder(feature, importance), y = importance)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        coord_flip() +
        labs(title = "Top 10 Feature Importance (Random Forest)",
             x = "Features", y = "Importance") +
        theme_minimal()
    }
  })
  
  # Risk level distribution
  output$risk_level_distribution <- renderPlot({
    if (!is.null(values$data) && "risk_score" %in% names(values$data)) {
      # Create risk levels
      risk_levels <- cut(values$data$risk_score, 
                        breaks = c(0, 0.3, 0.6, 0.8, 1), 
                        labels = c("Low", "Medium", "High", "Critical"),
                        include.lowest = TRUE)
      
      risk_level_df <- data.frame(risk_level = risk_levels)
      
      ggplot(risk_level_df, aes(x = risk_level, fill = risk_level)) +
        geom_bar() +
        scale_fill_brewer(type = "seq", palette = "Reds") +
        labs(title = "Risk Level Distribution",
             x = "Risk Level", y = "Count") +
        theme_minimal() +
        theme(legend.position = "none")
    }
  })
  
  # Amount vs Risk Score scatter plot
  output$amount_risk_scatter <- renderPlot({
    if (!is.null(values$data) && "amount" %in% names(values$data) && "risk_score" %in% names(values$data)) {
      ggplot(values$data, aes(x = amount, y = risk_score)) +
        geom_point(alpha = 0.6, color = "steelblue") +
        geom_smooth(method = "lm", color = "red") +
        scale_x_log10() +
        labs(title = "Amount vs Risk Score",
             x = "Amount (log scale)", y = "Risk Score") +
        theme_minimal()
    }
  })
  
  # Model comparison plot
  output$model_comparison <- renderPlot({
    if (!is.null(values$performance)) {
      # Create comparison data frame
      comparison_df <- data.frame(
        model = names(values$performance),
        rmse = sapply(values$performance, function(x) x$rmse),
        r_squared = sapply(values$performance, function(x) x$r_squared)
      )
      
      # Plot RMSE comparison
      p1 <- ggplot(comparison_df, aes(x = reorder(model, rmse), y = rmse)) +
        geom_bar(stat = "identity", fill = "steelblue") +
        coord_flip() +
        labs(title = "Model RMSE Comparison", x = "Model", y = "RMSE") +
        theme_minimal()
      
      # Plot R-squared comparison
      p2 <- ggplot(comparison_df, aes(x = reorder(model, r_squared), y = r_squared)) +
        geom_bar(stat = "identity", fill = "green") +
        coord_flip() +
        labs(title = "Model R-squared Comparison", x = "Model", y = "R-squared") +
        theme_minimal()
      
      grid.arrange(p1, p2, ncol = 2)
    }
  })
  
  # Prediction vs Actual plot
  output$prediction_actual <- renderPlot({
    if (!is.null(values$predictions) && !is.null(values$models)) {
      # Use ensemble prediction
      ensemble_pred <- values$predictions$ensemble_prediction
      test_target <- values$models$test_target
      
      prediction_df <- data.frame(
        actual = test_target,
        predicted = ensemble_pred
      )
      
      ggplot(prediction_df, aes(x = actual, y = predicted)) +
        geom_point(alpha = 0.6, color = "steelblue") +
        geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
        labs(title = "Prediction vs Actual (Ensemble Model)",
             x = "Actual Risk Score", y = "Predicted Risk Score") +
        theme_minimal()
    }
  })
  
  # Missing values plot
  output$missing_values_plot <- renderPlot({
    if (!is.null(values$data)) {
      # Create missing values visualization
      missing_data <- VIM::aggr(values$data, col = c('navyblue', 'red'), 
                               numbers = TRUE, sortVars = TRUE)
      plot(missing_data, main = "Missing Values Analysis")
    }
  })
  
  # Outlier detection plot
  output$outlier_plot <- renderPlot({
    if (!is.null(values$data)) {
      numeric_data <- values$data[, sapply(values$data, is.numeric)]
      if (ncol(numeric_data) > 0) {
        # Box plot for outlier detection
        boxplot(numeric_data, main = "Outlier Detection", 
                col = "lightblue", las = 2)
      }
    }
  })
}

# Run the Shiny application
shinyApp(ui = ui, server = server)
