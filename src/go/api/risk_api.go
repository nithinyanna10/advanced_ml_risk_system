/**
 * Advanced Go API for Financial Risk Assessment
 * High-performance microservice with concurrent processing
 * 
 * Technologies: Go, Gin, GORM, Redis, Kafka, Prometheus
 * Author: Nithin Yanna
 * Date: 2025
 */

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gin-contrib/cors"
	"github.com/gin-contrib/pprof"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"gorm.io/gorm"
	"gorm.io/driver/postgres"
	"github.com/go-redis/redis/v8"
	"github.com/segmentio/kafka-go"
)

// RiskAssessmentRequest represents a risk assessment request
type RiskAssessmentRequest struct {
	TransactionID string    `json:"transaction_id" binding:"required"`
	CustomerID    string    `json:"customer_id" binding:"required"`
	Amount        float64   `json:"amount" binding:"required"`
	Timestamp     time.Time `json:"timestamp"`
	Location      string    `json:"location"`
	MerchantID    string    `json:"merchant_id"`
	CustomerAge   int       `json:"customer_age"`
	PaymentMethod string    `json:"payment_method"`
}

// RiskAssessmentResponse represents the risk assessment response
type RiskAssessmentResponse struct {
	TransactionID     string    `json:"transaction_id"`
	RiskScore         float64   `json:"risk_score"`
	RiskLevel         string    `json:"risk_level"`
	Confidence        float64   `json:"confidence"`
	RecommendedAction string    `json:"recommended_action"`
	ProcessingTime    int64     `json:"processing_time_ms"`
	Timestamp         time.Time `json:"timestamp"`
	Features          map[string]interface{} `json:"features,omitempty"`
	Explanation       map[string]interface{} `json:"explanation,omitempty"`
}

// RiskStatistics represents risk statistics
type RiskStatistics struct {
	TotalAssessments   int64   `json:"total_assessments"`
	HighRiskCount      int64   `json:"high_risk_count"`
	MediumRiskCount    int64   `json:"medium_risk_count"`
	LowRiskCount       int64   `json:"low_risk_count"`
	AverageRiskScore   float64 `json:"average_risk_score"`
	ProcessingTimeMs   float64 `json:"average_processing_time_ms"`
}

// RiskAPI represents the main API struct
type RiskAPI struct {
	db          *gorm.DB
	redis       *redis.Client
	kafkaWriter *kafka.Writer
	mlService   *MLService
	metrics     *Metrics
	config      *Config
}

// Config represents application configuration
type Config struct {
	DatabaseURL    string
	RedisURL       string
	KafkaBrokers   []string
	MLServiceURL   string
	Port           string
	Workers        int
	CacheTTL       time.Duration
}

// Metrics represents Prometheus metrics
type Metrics struct {
	RequestsTotal     prometheus.Counter
	RequestDuration   prometheus.Histogram
	RiskScoreGauge    prometheus.Gauge
	ActiveConnections prometheus.Gauge
	ErrorRate         prometheus.Counter
}

// MLService represents the ML service client
type MLService struct {
	client *http.Client
	url    string
}

// NewRiskAPI creates a new RiskAPI instance
func NewRiskAPI(config *Config) (*RiskAPI, error) {
	// Initialize database
	db, err := gorm.Open(postgres.Open(config.DatabaseURL), &gorm.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	// Initialize Redis
	rdb := redis.NewClient(&redis.Options{
		Addr: config.RedisURL,
	})

	// Initialize Kafka writer
	kafkaWriter := &kafka.Writer{
		Addr:    kafka.TCP(config.KafkaBrokers...),
		Topic:   "risk-assessments",
		Balancer: &kafka.LeastBytes{},
	}

	// Initialize ML service
	mlService := &MLService{
		client: &http.Client{Timeout: 5 * time.Second},
		url:    config.MLServiceURL,
	}

	// Initialize metrics
	metrics := &Metrics{
		RequestsTotal: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "risk_api_requests_total",
			Help: "Total number of risk assessment requests",
		}),
		RequestDuration: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "risk_api_request_duration_seconds",
			Help:    "Duration of risk assessment requests",
			Buckets: prometheus.DefBuckets,
		}),
		RiskScoreGauge: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "risk_api_current_risk_score",
			Help: "Current average risk score",
		}),
		ActiveConnections: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "risk_api_active_connections",
			Help: "Number of active connections",
		}),
		ErrorRate: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "risk_api_errors_total",
			Help: "Total number of errors",
		}),
	}

	// Register metrics
	prometheus.MustRegister(
		metrics.RequestsTotal,
		metrics.RequestDuration,
		metrics.RiskScoreGauge,
		metrics.ActiveConnections,
		metrics.ErrorRate,
	)

	return &RiskAPI{
		db:          db,
		redis:       rdb,
		kafkaWriter: kafkaWriter,
		mlService:   mlService,
		metrics:     metrics,
		config:      config,
	}, nil
}

// AssessRisk handles risk assessment requests
func (api *RiskAPI) AssessRisk(c *gin.Context) {
	start := time.Now()
	api.metrics.RequestsTotal.Inc()
	api.metrics.ActiveConnections.Inc()
	defer func() {
		api.metrics.ActiveConnections.Dec()
		api.metrics.RequestDuration.Observe(time.Since(start).Seconds())
	}()

	var req RiskAssessmentRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		api.metrics.ErrorRate.Inc()
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Set timestamp if not provided
	if req.Timestamp.IsZero() {
		req.Timestamp = time.Now()
	}

	// Check cache first
	cacheKey := fmt.Sprintf("risk:%s", req.TransactionID)
	cached, err := api.getFromCache(cacheKey)
	if err == nil && cached != nil {
		c.JSON(http.StatusOK, cached)
		return
	}

	// Perform risk assessment
	response, err := api.performRiskAssessment(req)
	if err != nil {
		api.metrics.ErrorRate.Inc()
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Cache the result
	api.setCache(cacheKey, response, api.config.CacheTTL)

	// Publish to Kafka
	go api.publishToKafka(response)

	c.JSON(http.StatusOK, response)
}

// BatchAssessRisk handles batch risk assessment
func (api *RiskAPI) BatchAssessRisk(c *gin.Context) {
	start := time.Now()
	api.metrics.RequestsTotal.Inc()
	defer api.metrics.RequestDuration.Observe(time.Since(start).Seconds())

	var requests []RiskAssessmentRequest
	if err := c.ShouldBindJSON(&requests); err != nil {
		api.metrics.ErrorRate.Inc()
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	// Process requests concurrently
	responses := make([]RiskAssessmentResponse, len(requests))
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, api.config.Workers)

	for i, req := range requests {
		wg.Add(1)
		go func(index int, request RiskAssessmentRequest) {
			defer wg.Done()
			semaphore <- struct{}{} // Acquire semaphore
			defer func() { <-semaphore }() // Release semaphore

			response, err := api.performRiskAssessment(request)
			if err != nil {
				api.metrics.ErrorRate.Inc()
				response = RiskAssessmentResponse{
					TransactionID: request.TransactionID,
					RiskScore:     0.5, // Default risk score
					RiskLevel:     "MEDIUM",
					Confidence:    0.5,
					Timestamp:     time.Now(),
				}
			}
			responses[index] = response
		}(i, req)
	}

	wg.Wait()
	c.JSON(http.StatusOK, responses)
}

// GetRiskStatistics returns risk statistics
func (api *RiskAPI) GetRiskStatistics(c *gin.Context) {
	// Get time range from query parameters
	startTimeStr := c.Query("start_time")
	endTimeStr := c.Query("end_time")

	var startTime, endTime time.Time
	var err error

	if startTimeStr != "" {
		startTime, err = time.Parse(time.RFC3339, startTimeStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid start_time format"})
			return
		}
	} else {
		startTime = time.Now().Add(-24 * time.Hour)
	}

	if endTimeStr != "" {
		endTime, err = time.Parse(time.RFC3339, endTimeStr)
		if err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid end_time format"})
			return
		}
	} else {
		endTime = time.Now()
	}

	// Get statistics from database
	var stats RiskStatistics
	err = api.db.Raw(`
		SELECT 
			COUNT(*) as total_assessments,
			COUNT(CASE WHEN risk_level = 'HIGH' THEN 1 END) as high_risk_count,
			COUNT(CASE WHEN risk_level = 'MEDIUM' THEN 1 END) as medium_risk_count,
			COUNT(CASE WHEN risk_level = 'LOW' THEN 1 END) as low_risk_count,
			AVG(risk_score) as average_risk_score,
			AVG(processing_time_ms) as average_processing_time_ms
		FROM risk_assessments 
		WHERE timestamp BETWEEN ? AND ?
	`, startTime, endTime).Scan(&stats).Error

	if err != nil {
		api.metrics.ErrorRate.Inc()
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, stats)
}

// GetHealth returns health status
func (api *RiskAPI) GetHealth(c *gin.Context) {
	// Check database connection
	var dbStatus string
	if err := api.db.Raw("SELECT 1").Error; err != nil {
		dbStatus = "unhealthy"
	} else {
		dbStatus = "healthy"
	}

	// Check Redis connection
	var redisStatus string
	if err := api.redis.Ping(context.Background()).Err(); err != nil {
		redisStatus = "unhealthy"
	} else {
		redisStatus = "healthy"
	}

	health := gin.H{
		"status":    "healthy",
		"timestamp": time.Now(),
		"services": gin.H{
			"database": dbStatus,
			"redis":    redisStatus,
		},
	}

	statusCode := http.StatusOK
	if dbStatus != "healthy" || redisStatus != "healthy" {
		statusCode = http.StatusServiceUnavailable
		health["status"] = "unhealthy"
	}

	c.JSON(statusCode, health)
}

// performRiskAssessment performs the actual risk assessment
func (api *RiskAPI) performRiskAssessment(req RiskAssessmentRequest) (*RiskAssessmentResponse, error) {
	start := time.Now()

	// Feature engineering
	features := api.engineerFeatures(req)

	// Call ML service
	riskScore, confidence, err := api.mlService.PredictRisk(features)
	if err != nil {
		// Fallback to rule-based assessment
		riskScore = api.ruleBasedRiskAssessment(req)
		confidence = 0.7
	}

	// Determine risk level
	riskLevel := api.determineRiskLevel(riskScore)
	recommendedAction := api.determineRecommendedAction(riskLevel)

	// Create response
	response := &RiskAssessmentResponse{
		TransactionID:     req.TransactionID,
		RiskScore:         riskScore,
		RiskLevel:         riskLevel,
		Confidence:        confidence,
		RecommendedAction: recommendedAction,
		ProcessingTime:    time.Since(start).Milliseconds(),
		Timestamp:         time.Now(),
		Features:          features,
	}

	// Update metrics
	api.metrics.RiskScoreGauge.Set(riskScore)

	// Store in database
	go api.storeRiskAssessment(response)

	return response, nil
}

// engineerFeatures creates features for ML model
func (api *RiskAPI) engineerFeatures(req RiskAssessmentRequest) map[string]interface{} {
	features := make(map[string]interface{})

	// Basic features
	features["amount"] = req.Amount
	features["customer_age"] = req.CustomerAge
	features["amount_log"] = math.Log(req.Amount + 1)
	features["amount_sqrt"] = math.Sqrt(req.Amount)

	// Time-based features
	hour := req.Timestamp.Hour()
	dayOfWeek := int(req.Timestamp.Weekday())
	features["hour"] = hour
	features["day_of_week"] = dayOfWeek
	features["is_weekend"] = dayOfWeek == 0 || dayOfWeek == 6
	features["is_night"] = hour >= 22 || hour <= 6

	// Customer features (from cache or database)
	customerFeatures := api.getCustomerFeatures(req.CustomerID)
	features["customer_transaction_count"] = customerFeatures["transaction_count"]
	features["customer_avg_amount"] = customerFeatures["avg_amount"]
	features["customer_risk_history"] = customerFeatures["risk_history"]

	// Location features
	locationFeatures := api.getLocationFeatures(req.Location)
	features["location_risk_score"] = locationFeatures["risk_score"]
	features["location_transaction_count"] = locationFeatures["transaction_count"]

	return features
}

// ruleBasedRiskAssessment provides fallback risk assessment
func (api *RiskAPI) ruleBasedRiskAssessment(req RiskAssessmentRequest) float64 {
	riskScore := 0.0

	// Amount-based risk
	if req.Amount > 10000 {
		riskScore += 0.3
	} else if req.Amount > 5000 {
		riskScore += 0.2
	}

	// Time-based risk
	hour := req.Timestamp.Hour()
	if hour >= 22 || hour <= 6 {
		riskScore += 0.2
	}

	// Weekend risk
	if req.Timestamp.Weekday() == 0 || req.Timestamp.Weekday() == 6 {
		riskScore += 0.1
	}

	// Customer age risk
	if req.CustomerAge < 25 || req.CustomerAge > 65 {
		riskScore += 0.1
	}

	return math.Min(1.0, riskScore)
}

// determineRiskLevel determines risk level based on score
func (api *RiskAPI) determineRiskLevel(riskScore float64) string {
	if riskScore >= 0.8 {
		return "HIGH"
	} else if riskScore >= 0.5 {
		return "MEDIUM"
	}
	return "LOW"
}

// determineRecommendedAction determines recommended action
func (api *RiskAPI) determineRecommendedAction(riskLevel string) string {
	switch riskLevel {
	case "HIGH":
		return "BLOCK"
	case "MEDIUM":
		return "REVIEW"
	default:
		return "APPROVE"
	}
}

// getCustomerFeatures retrieves customer features
func (api *RiskAPI) getCustomerFeatures(customerID string) map[string]interface{} {
	// Try cache first
	cacheKey := fmt.Sprintf("customer:%s", customerID)
	cached, err := api.getFromCache(cacheKey)
	if err == nil && cached != nil {
		return cached.(map[string]interface{})
	}

	// Query database
	var result map[string]interface{}
	api.db.Raw(`
		SELECT 
			COUNT(*) as transaction_count,
			AVG(amount) as avg_amount,
			AVG(risk_score) as risk_history
		FROM risk_assessments 
		WHERE customer_id = ? 
		AND timestamp > NOW() - INTERVAL '30 days'
	`, customerID).Scan(&result)

	// Cache result
	api.setCache(cacheKey, result, 5*time.Minute)

	return result
}

// getLocationFeatures retrieves location features
func (api *RiskAPI) getLocationFeatures(location string) map[string]interface{} {
	// Try cache first
	cacheKey := fmt.Sprintf("location:%s", location)
	cached, err := api.getFromCache(cacheKey)
	if err == nil && cached != nil {
		return cached.(map[string]interface{})
	}

	// Query database
	var result map[string]interface{}
	api.db.Raw(`
		SELECT 
			COUNT(*) as transaction_count,
			AVG(risk_score) as risk_score
		FROM risk_assessments 
		WHERE location = ? 
		AND timestamp > NOW() - INTERVAL '7 days'
	`, location).Scan(&result)

	// Cache result
	api.setCache(cacheKey, result, 10*time.Minute)

	return result
}

// Cache operations
func (api *RiskAPI) getFromCache(key string) (interface{}, error) {
	val, err := api.redis.Get(context.Background(), key).Result()
	if err != nil {
		return nil, err
	}

	var result interface{}
	err = json.Unmarshal([]byte(val), &result)
	return result, err
}

func (api *RiskAPI) setCache(key string, value interface{}, ttl time.Duration) {
	data, err := json.Marshal(value)
	if err != nil {
		return
	}
	api.redis.Set(context.Background(), key, data, ttl)
}

// Kafka operations
func (api *RiskAPI) publishToKafka(response *RiskAssessmentResponse) {
	data, err := json.Marshal(response)
	if err != nil {
		log.Printf("Error marshaling response: %v", err)
		return
	}

	err = api.kafkaWriter.WriteMessages(context.Background(),
		kafka.Message{
			Key:   []byte(response.TransactionID),
			Value: data,
		},
	)
	if err != nil {
		log.Printf("Error publishing to Kafka: %v", err)
	}
}

// Database operations
func (api *RiskAPI) storeRiskAssessment(response *RiskAssessmentResponse) {
	// Implementation to store in database
	// This would typically involve creating a struct and using GORM
}

// MLService methods
func (ml *MLService) PredictRisk(features map[string]interface{}) (float64, float64, error) {
	// Implementation to call ML service
	// This would make an HTTP request to the ML service
	return 0.5, 0.8, nil // Placeholder
}

// Main function
func main() {
	// Load configuration
	config := &Config{
		DatabaseURL:  getEnv("DATABASE_URL", "postgres://user:password@localhost/riskdb"),
		RedisURL:     getEnv("REDIS_URL", "localhost:6379"),
		KafkaBrokers: []string{getEnv("KAFKA_BROKERS", "localhost:9092")},
		MLServiceURL: getEnv("ML_SERVICE_URL", "http://localhost:8081"),
		Port:         getEnv("PORT", "8080"),
		Workers:      getEnvInt("WORKERS", 10),
		CacheTTL:     5 * time.Minute,
	}

	// Create API instance
	api, err := NewRiskAPI(config)
	if err != nil {
		log.Fatal("Failed to create API instance:", err)
	}

	// Setup routes
	router := gin.Default()
	router.Use(cors.Default())

	// Add pprof for profiling
	pprof.Register(router)

	// API routes
	apiGroup := router.Group("/api/v1")
	{
		apiGroup.POST("/assess", api.AssessRisk)
		apiGroup.POST("/assess/batch", api.BatchAssessRisk)
		apiGroup.GET("/statistics", api.GetRiskStatistics)
		apiGroup.GET("/health", api.GetHealth)
	}

	// Metrics endpoint
	router.GET("/metrics", gin.WrapH(promhttp.Handler()))

	// Start server
	log.Printf("Starting server on port %s", config.Port)
	log.Fatal(router.Run(":" + config.Port))
}

// Utility functions
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}
