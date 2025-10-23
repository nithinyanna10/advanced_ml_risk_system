/**
 * Advanced Spark Data Processing Pipeline
 * Large-scale data processing for financial risk assessment
 * 
 * Technologies: Apache Spark, Scala, Delta Lake, MLlib
 * Author: Nithin Yanna
 * Date: 2025
 */

package com.advancedml.risksystem.spark

import org.apache.spark.sql.{SparkSession, DataFrame, Dataset}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder}
import org.apache.spark.ml.classification.{RandomForestClassifier, GBTClassifier, LogisticRegression}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.streaming.{StreamingQuery, Trigger}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.storage.StorageLevel

import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import scala.util.{Try, Success, Failure}
import scala.collection.mutable

/**
 * Advanced Spark-based data processing pipeline for financial risk assessment
 * Features:
 * - Real-time streaming data processing
 * - Advanced feature engineering
 * - Machine learning model training and inference
 * - Data quality monitoring
 * - Performance optimization
 */
class DataProcessingPipeline(spark: SparkSession) {
  
  import spark.implicits._
  
  // Configuration
  private val config = PipelineConfig()
  private val logger = org.slf4j.LoggerFactory.getLogger(this.getClass)
  
  // Data quality metrics
  private val dataQualityMetrics = mutable.Map[String, DataQualityMetrics]()
  
  /**
   * Main data processing pipeline
   */
  def processData(inputPath: String, outputPath: String): Unit = {
    logger.info("Starting data processing pipeline")
    
    try {
      // Step 1: Data ingestion
      val rawData = ingestData(inputPath)
      
      // Step 2: Data quality checks
      val qualityReport = performDataQualityChecks(rawData)
      logger.info(s"Data quality report: $qualityReport")
      
      // Step 3: Data cleaning and preprocessing
      val cleanedData = cleanAndPreprocessData(rawData)
      
      // Step 4: Feature engineering
      val featuresData = engineerFeatures(cleanedData)
      
      // Step 5: Data validation
      val validatedData = validateData(featuresData)
      
      // Step 6: Save processed data
      saveProcessedData(validatedData, outputPath)
      
      logger.info("Data processing pipeline completed successfully")
      
    } catch {
      case e: Exception =>
        logger.error("Error in data processing pipeline", e)
        throw new DataProcessingException("Pipeline processing failed", e)
    }
  }
  
  /**
   * Real-time streaming data processing
   */
  def processStreamingData(kafkaBootstrapServers: String, 
                          topics: Array[String]): StreamingQuery = {
    logger.info("Starting streaming data processing")
    
    val streamingDF = spark
      .readStream
      .format("kafka")
      .option("kafka.bootstrap.servers", kafkaBootstrapServers)
      .option("subscribe", topics.mkString(","))
      .option("startingOffsets", "latest")
      .load()
    
    // Parse JSON data
    val parsedDF = streamingDF
      .select(
        from_json($"value".cast("string"), getTransactionSchema()).as("data")
      )
      .select("data.*")
    
    // Process streaming data
    val processedDF = parsedDF
      .withColumn("processing_timestamp", current_timestamp())
      .withColumn("risk_score", calculateRiskScoreUDF($"amount", $"customer_id", $"location"))
      .withColumn("risk_level", 
        when($"risk_score" >= 0.8, "HIGH")
        .when($"risk_score" >= 0.5, "MEDIUM")
        .otherwise("LOW"))
    
    // Write to Delta Lake
    val query = processedDF
      .writeStream
      .format("delta")
      .option("checkpointLocation", "/tmp/checkpoint")
      .outputMode("append")
      .trigger(Trigger.ProcessingTime("10 seconds"))
      .start("/data/processed/risk_assessments")
    
    logger.info("Streaming data processing started")
    query
  }
  
  /**
   * Ingest data from various sources
   */
  private def ingestData(inputPath: String): DataFrame = {
    logger.info(s"Ingesting data from: $inputPath")
    
    val data = spark.read
      .format("delta")
      .option("mergeSchema", "true")
      .load(inputPath)
    
    // Cache for performance
    data.persist(StorageLevel.MEMORY_AND_DISK_SER)
    
    logger.info(s"Ingested ${data.count()} records")
    data
  }
  
  /**
   * Perform comprehensive data quality checks
   */
  private def performDataQualityChecks(data: DataFrame): DataQualityReport = {
    logger.info("Performing data quality checks")
    
    val totalRecords = data.count()
    val nullCounts = data.columns.map(col => 
      (col, data.filter(col(col).isNull).count())
    ).toMap
    
    val duplicateCount = data.count() - data.distinct().count()
    
    val dataTypes = data.dtypes.toMap
    
    val qualityMetrics = DataQualityMetrics(
      totalRecords = totalRecords,
      nullCounts = nullCounts,
      duplicateCount = duplicateCount,
      dataTypes = dataTypes,
      completeness = (totalRecords - nullCounts.values.sum) / totalRecords.toDouble,
      uniqueness = (totalRecords - duplicateCount) / totalRecords.toDouble
    )
    
    dataQualityMetrics("main_dataset") = qualityMetrics
    
    DataQualityReport(
      datasetName = "main_dataset",
      metrics = qualityMetrics,
      timestamp = LocalDateTime.now(),
      passed = qualityMetrics.completeness > 0.95 && qualityMetrics.uniqueness > 0.99
    )
  }
  
  /**
   * Clean and preprocess data
   */
  private def cleanAndPreprocessData(data: DataFrame): DataFrame = {
    logger.info("Cleaning and preprocessing data")
    
    data
      // Remove duplicates
      .dropDuplicates()
      
      // Handle null values
      .na.fill(Map(
        "amount" -> 0.0,
        "customer_age" -> 30,
        "transaction_count" -> 0
      ))
      
      // Data type conversions
      .withColumn("amount", $"amount".cast(DoubleType))
      .withColumn("customer_age", $"customer_age".cast(IntegerType))
      .withColumn("transaction_count", $"transaction_count".cast(IntegerType))
      
      // Feature transformations
      .withColumn("amount_log", log($"amount" + 1))
      .withColumn("amount_sqrt", sqrt($"amount"))
      .withColumn("is_weekend", 
        when(dayofweek($"timestamp") === 1 || dayofweek($"timestamp") === 7, 1)
        .otherwise(0))
      .withColumn("is_night", 
        when(hour($"timestamp") >= 22 || hour($"timestamp") <= 6, 1)
        .otherwise(0))
      
      // Outlier detection and treatment
      .withColumn("amount_zscore", 
        (($"amount" - avg($"amount").over()) / stddev($"amount").over()))
      .filter($"amount_zscore" < 3) // Remove extreme outliers
  }
  
  /**
   * Advanced feature engineering
   */
  private def engineerFeatures(data: DataFrame): DataFrame = {
    logger.info("Engineering features")
    
    // Customer-level aggregations
    val customerFeatures = data
      .groupBy("customer_id")
      .agg(
        count("*").as("total_transactions"),
        sum("amount").as("total_amount"),
        avg("amount").as("avg_amount"),
        max("amount").as("max_amount"),
        min("amount").as("min_amount"),
        stddev("amount").as("amount_std"),
        countDistinct("merchant_id").as("unique_merchants"),
        countDistinct("location").as("unique_locations")
      )
    
    // Time-based features
    val timeFeatures = data
      .withColumn("hour", hour($"timestamp"))
      .withColumn("day_of_week", dayofweek($"timestamp"))
      .withColumn("month", month($"timestamp"))
      .withColumn("is_weekend", 
        when($"day_of_week" === 1 || $"day_of_week" === 7, 1).otherwise(0))
    
    // Location-based features
    val locationFeatures = data
      .groupBy("location")
      .agg(
        count("*").as("location_transaction_count"),
        avg("amount").as("location_avg_amount")
      )
    
    // Join all features
    val featuresData = data
      .join(customerFeatures, "customer_id")
      .join(locationFeatures, "location")
      .withColumn("amount_ratio", $"amount" / $"avg_amount")
      .withColumn("location_risk_score", 
        when($"location_transaction_count" < 10, 0.8)
        .when($"location_transaction_count" < 100, 0.5)
        .otherwise(0.2))
    
    featuresData
  }
  
  /**
   * Data validation
   */
  private def validateData(data: DataFrame): DataFrame = {
    logger.info("Validating processed data")
    
    // Business rule validations
    val validatedData = data
      .filter($"amount" > 0) // Positive amounts only
      .filter($"customer_age" >= 18 && $"customer_age" <= 100) // Valid age range
      .filter($"amount" <= 1000000) // Reasonable amount limit
      .filter($"risk_score" >= 0 && $"risk_score" <= 1) // Valid risk score range
    
    logger.info(s"Data validation completed. ${validatedData.count()} valid records")
    validatedData
  }
  
  /**
   * Save processed data
   */
  private def saveProcessedData(data: DataFrame, outputPath: String): Unit = {
    logger.info(s"Saving processed data to: $outputPath")
    
    data.write
      .format("delta")
      .mode("overwrite")
      .option("mergeSchema", "true")
      .save(outputPath)
    
    logger.info("Data saved successfully")
  }
  
  /**
   * Train machine learning models
   */
  def trainMLModels(trainingData: DataFrame): PipelineModel = {
    logger.info("Training machine learning models")
    
    // Prepare features
    val featureCols = Array("amount", "customer_age", "transaction_count", 
                           "total_transactions", "avg_amount", "amount_ratio")
    
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
    
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaled_features")
    
    // Create ensemble of models
    val rf = new RandomForestClassifier()
      .setLabelCol("risk_label")
      .setFeaturesCol("scaled_features")
      .setNumTrees(100)
      .setMaxDepth(10)
    
    val gbt = new GBTClassifier()
      .setLabelCol("risk_label")
      .setFeaturesCol("scaled_features")
      .setMaxIter(100)
    
    val lr = new LogisticRegression()
      .setLabelCol("risk_label")
      .setFeaturesCol("scaled_features")
      .setMaxIter(100)
    
    // Create pipeline
    val pipeline = new Pipeline()
      .setStages(Array(assembler, scaler, rf))
    
    // Train model
    val model = pipeline.fit(trainingData)
    
    logger.info("Model training completed")
    model
  }
  
  /**
   * Hyperparameter tuning with cross-validation
   */
  def tuneHyperparameters(trainingData: DataFrame): PipelineModel = {
    logger.info("Starting hyperparameter tuning")
    
    val featureCols = Array("amount", "customer_age", "transaction_count")
    val assembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")
    
    val rf = new RandomForestClassifier()
      .setLabelCol("risk_label")
      .setFeaturesCol("features")
    
    val pipeline = new Pipeline()
      .setStages(Array(assembler, rf))
    
    // Parameter grid
    val paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(50, 100, 200))
      .addGrid(rf.maxDepth, Array(5, 10, 15))
      .addGrid(rf.maxBins, Array(32, 64, 128))
      .build()
    
    // Cross-validation
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("risk_label")
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")
    
    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(4)
    
    val cvModel = cv.fit(trainingData)
    
    logger.info("Hyperparameter tuning completed")
    cvModel.bestModel.asInstanceOf[PipelineModel]
  }
  
  /**
   * Real-time model inference
   */
  def predictRisk(model: PipelineModel, data: DataFrame): DataFrame = {
    logger.info("Performing risk prediction")
    
    val predictions = model.transform(data)
    
    predictions
      .withColumn("risk_level",
        when($"probability"(1) >= 0.8, "HIGH")
        .when($"probability"(1) >= 0.5, "MEDIUM")
        .otherwise("LOW"))
      .withColumn("confidence", $"probability"(1))
  }
  
  /**
   * Calculate risk score UDF
   */
  private def calculateRiskScoreUDF = udf { (amount: Double, customerId: String, location: String) =>
    // Simplified risk calculation
    val baseScore = if (amount > 10000) 0.7 else 0.3
    val customerRisk = if (customerId.startsWith("NEW")) 0.5 else 0.2
    val locationRisk = if (location.contains("HIGH_RISK")) 0.6 else 0.1
    
    Math.min(1.0, baseScore + customerRisk + locationRisk)
  }
  
  /**
   * Get transaction schema
   */
  private def getTransactionSchema(): StructType = {
    StructType(Array(
      StructField("transaction_id", StringType, nullable = false),
      StructField("customer_id", StringType, nullable = false),
      StructField("amount", DoubleType, nullable = false),
      StructField("timestamp", TimestampType, nullable = false),
      StructField("location", StringType, nullable = true),
      StructField("merchant_id", StringType, nullable = true),
      StructField("customer_age", IntegerType, nullable = true)
    ))
  }
  
  /**
   * Performance monitoring
   */
  def monitorPerformance(): Unit = {
    logger.info("Monitoring pipeline performance")
    
    val metrics = spark.sparkContext.statusTracker.getExecutorInfos
    logger.info(s"Active executors: ${metrics.length}")
    
    // Log memory usage
    val memoryUsed = spark.sparkContext.statusTracker.getExecutorInfos
      .map(_.memoryUsed).sum
    logger.info(s"Total memory used: ${memoryUsed / 1024 / 1024} MB")
  }
}

/**
 * Configuration case class
 */
case class PipelineConfig(
  batchSize: Int = 10000,
  checkpointInterval: Int = 10,
  maxRetries: Int = 3,
  timeoutMinutes: Int = 30
)

/**
 * Data quality metrics
 */
case class DataQualityMetrics(
  totalRecords: Long,
  nullCounts: Map[String, Long],
  duplicateCount: Long,
  dataTypes: Map[String, String],
  completeness: Double,
  uniqueness: Double
)

/**
 * Data quality report
 */
case class DataQualityReport(
  datasetName: String,
  metrics: DataQualityMetrics,
  timestamp: LocalDateTime,
  passed: Boolean
)

/**
 * Custom exception
 */
class DataProcessingException(message: String, cause: Throwable) 
  extends RuntimeException(message, cause)

/**
 * Main application
 */
object DataProcessingPipelineApp {
  
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Advanced Data Processing Pipeline")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .getOrCreate()
    
    try {
      val pipeline = new DataProcessingPipeline(spark)
      
      // Process batch data
      pipeline.processData("/data/raw/transactions", "/data/processed/transactions")
      
      // Start streaming processing
      val streamingQuery = pipeline.processStreamingData(
        "localhost:9092", 
        Array("transactions", "risk-events")
      )
      
      // Wait for streaming to complete
      streamingQuery.awaitTermination()
      
    } finally {
      spark.stop()
    }
  }
}
