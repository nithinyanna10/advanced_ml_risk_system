/**
 * Advanced Rust Risk Engine
 * Memory-safe, high-performance risk assessment engine
 * 
 * Technologies: Rust, Tokio, Serde, Diesel, Redis, Kafka
 * Author: Nithin Yanna
 * Date: 2025
 */

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tokio::time::interval;
use anyhow::{Result, Context};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::models::*;
use crate::ml::*;
use crate::cache::*;
use crate::database::*;
use crate::kafka::*;

/// Main risk assessment engine
#[derive(Clone)]
pub struct RiskEngine {
    ml_service: Arc<MLService>,
    cache: Arc<CacheService>,
    database: Arc<DatabaseService>,
    kafka_producer: Arc<KafkaProducer>,
    config: RiskEngineConfig,
    metrics: Arc<RwLock<EngineMetrics>>,
}

/// Risk assessment request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentRequest {
    pub transaction_id: Uuid,
    pub customer_id: String,
    pub amount: f64,
    pub timestamp: DateTime<Utc>,
    pub location: Option<String>,
    pub merchant_id: Option<String>,
    pub customer_age: Option<u32>,
    pub payment_method: Option<String>,
    pub device_fingerprint: Option<String>,
    pub ip_address: Option<String>,
}

/// Risk assessment response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessmentResponse {
    pub transaction_id: Uuid,
    pub risk_score: f64,
    pub risk_level: RiskLevel,
    pub confidence: f64,
    pub recommended_action: RecommendedAction,
    pub processing_time_ms: u64,
    pub timestamp: DateTime<Utc>,
    pub features: HashMap<String, f64>,
    pub explanation: Option<Explanation>,
    pub business_rules_applied: Vec<String>,
}

/// Risk levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Recommended actions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RecommendedAction {
    Approve,
    Review,
    Block,
    Escalate,
}

/// Model explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Explanation {
    pub feature_importance: HashMap<String, f64>,
    pub shap_values: HashMap<String, f64>,
    pub lime_explanation: Option<Vec<(String, f64)>>,
    pub decision_path: Vec<String>,
}

/// Engine configuration
#[derive(Debug, Clone)]
pub struct RiskEngineConfig {
    pub max_concurrent_requests: usize,
    pub cache_ttl_seconds: u64,
    pub model_timeout_ms: u64,
    pub enable_explanations: bool,
    pub enable_bias_detection: bool,
    pub fallback_threshold: f64,
}

/// Engine metrics
#[derive(Debug, Default)]
pub struct EngineMetrics {
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub average_processing_time_ms: f64,
    pub cache_hit_rate: f64,
    pub model_accuracy: f64,
    pub high_risk_percentage: f64,
}

impl RiskEngine {
    /// Create a new risk engine
    pub async fn new(
        ml_service: Arc<MLService>,
        cache: Arc<CacheService>,
        database: Arc<DatabaseService>,
        kafka_producer: Arc<KafkaProducer>,
        config: RiskEngineConfig,
    ) -> Result<Self> {
        let engine = Self {
            ml_service,
            cache,
            database,
            kafka_producer,
            config,
            metrics: Arc::new(RwLock::new(EngineMetrics::default())),
        };

        // Start background tasks
        engine.start_background_tasks().await?;

        Ok(engine)
    }

    /// Assess risk for a single transaction
    pub async fn assess_risk(&self, request: RiskAssessmentRequest) -> Result<RiskAssessmentResponse> {
        let start_time = Instant::now();
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_requests += 1;
        }

        // Check cache first
        let cache_key = format!("risk:{}", request.transaction_id);
        if let Some(cached_response) = self.cache.get(&cache_key).await? {
            let mut metrics = self.metrics.write().await;
            metrics.successful_requests += 1;
            metrics.cache_hit_rate = (metrics.cache_hit_rate * 0.9) + 0.1;
            return Ok(cached_response);
        }

        // Perform risk assessment
        let response = self.perform_risk_assessment(request).await?;
        
        // Cache the result
        self.cache.set(&cache_key, &response, Duration::from_secs(self.config.cache_ttl_seconds)).await?;

        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as u64;
        {
            let mut metrics = self.metrics.write().await;
            metrics.successful_requests += 1;
            metrics.average_processing_time_ms = 
                (metrics.average_processing_time_ms * 0.9) + (processing_time as f64 * 0.1);
            
            if response.risk_level == RiskLevel::High || response.risk_level == RiskLevel::Critical {
                metrics.high_risk_percentage = 
                    (metrics.high_risk_percentage * 0.9) + 0.1;
            }
        }

        // Publish to Kafka
        self.publish_risk_event(&response).await?;

        Ok(response)
    }

    /// Assess risk for multiple transactions (batch processing)
    pub async fn assess_risk_batch(&self, requests: Vec<RiskAssessmentRequest>) -> Result<Vec<RiskAssessmentResponse>> {
        let mut responses = Vec::with_capacity(requests.len());
        let semaphore = Arc::new(tokio::sync::Semaphore::new(self.config.max_concurrent_requests));

        let mut handles = Vec::new();

        for request in requests {
            let engine = self.clone();
            let semaphore = semaphore.clone();
            
            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                engine.assess_risk(request).await
            });
            
            handles.push(handle);
        }

        for handle in handles {
            match handle.await {
                Ok(Ok(response)) => responses.push(response),
                Ok(Err(e)) => {
                    let mut metrics = self.metrics.write().await;
                    metrics.failed_requests += 1;
                    eprintln!("Risk assessment failed: {}", e);
                },
                Err(e) => {
                    let mut metrics = self.metrics.write().await;
                    metrics.failed_requests += 1;
                    eprintln!("Task failed: {}", e);
                },
            }
        }

        Ok(responses)
    }

    /// Perform the actual risk assessment
    async fn perform_risk_assessment(&self, request: RiskAssessmentRequest) -> Result<RiskAssessmentResponse> {
        let start_time = Instant::now();

        // Feature engineering
        let features = self.engineer_features(&request).await?;

        // Get ML prediction
        let (risk_score, confidence) = match self.ml_service.predict_risk(&features).await {
            Ok((score, conf)) => (score, conf),
            Err(e) => {
                eprintln!("ML service failed, using fallback: {}", e);
                (self.fallback_risk_assessment(&request), 0.5)
            }
        };

        // Apply business rules
        let (final_risk_score, business_rules) = self.apply_business_rules(risk_score, &request);

        // Determine risk level and action
        let risk_level = self.determine_risk_level(final_risk_score);
        let recommended_action = self.determine_recommended_action(risk_level);

        // Generate explanation if enabled
        let explanation = if self.config.enable_explanations {
            self.generate_explanation(&features, final_risk_score).await.ok()
        } else {
            None
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(RiskAssessmentResponse {
            transaction_id: request.transaction_id,
            risk_score: final_risk_score,
            risk_level,
            confidence,
            recommended_action,
            processing_time_ms: processing_time,
            timestamp: Utc::now(),
            features,
            explanation,
            business_rules_applied: business_rules,
        })
    }

    /// Engineer features for ML model
    async fn engineer_features(&self, request: &RiskAssessmentRequest) -> Result<HashMap<String, f64>> {
        let mut features = HashMap::new();

        // Basic features
        features.insert("amount".to_string(), request.amount);
        features.insert("amount_log".to_string(), request.amount.ln());
        features.insert("amount_sqrt".to_string(), request.amount.sqrt());
        features.insert("amount_squared".to_string(), request.amount.powi(2));

        // Time-based features
        let hour = request.timestamp.hour() as f64;
        let day_of_week = request.timestamp.weekday().num_days_from_sunday() as f64;
        features.insert("hour".to_string(), hour);
        features.insert("day_of_week".to_string(), day_of_week);
        features.insert("is_weekend".to_string(), if day_of_week == 0.0 || day_of_week == 6.0 { 1.0 } else { 0.0 });
        features.insert("is_night".to_string(), if hour >= 22.0 || hour <= 6.0 { 1.0 } else { 0.0 });

        // Customer features
        if let Some(age) = request.customer_age {
            features.insert("customer_age".to_string(), age as f64);
            features.insert("is_young".to_string(), if age < 25 { 1.0 } else { 0.0 });
            features.insert("is_senior".to_string(), if age > 65 { 1.0 } else { 0.0 });
        }

        // Historical features
        let customer_features = self.get_customer_features(&request.customer_id).await?;
        features.insert("customer_transaction_count".to_string(), customer_features.transaction_count);
        features.insert("customer_avg_amount".to_string(), customer_features.avg_amount);
        features.insert("customer_risk_history".to_string(), customer_features.risk_history);

        // Location features
        if let Some(location) = &request.location {
            let location_features = self.get_location_features(location).await?;
            features.insert("location_risk_score".to_string(), location_features.risk_score);
            features.insert("location_transaction_count".to_string(), location_features.transaction_count);
        }

        Ok(features)
    }

    /// Apply business rules to risk score
    fn apply_business_rules(&self, base_score: f64, request: &RiskAssessmentRequest) -> (f64, Vec<String>) {
        let mut final_score = base_score;
        let mut applied_rules = Vec::new();

        // High amount threshold
        if request.amount > 10000.0 {
            final_score = final_score.max(0.7);
            applied_rules.push("HIGH_AMOUNT_THRESHOLD".to_string());
        }

        // New customer
        if request.customer_id.starts_with("NEW") {
            final_score = final_score.max(0.5);
            applied_rules.push("NEW_CUSTOMER".to_string());
        }

        // Unusual time
        let hour = request.timestamp.hour();
        if hour >= 22 || hour <= 6 {
            final_score = final_score.max(0.3);
            applied_rules.push("UNUSUAL_TIME".to_string());
        }

        // Weekend transactions
        if request.timestamp.weekday().num_days_from_sunday() == 0 || 
           request.timestamp.weekday().num_days_from_sunday() == 6 {
            final_score = final_score.max(0.2);
            applied_rules.push("WEEKEND_TRANSACTION".to_string());
        }

        // Age-based rules
        if let Some(age) = request.customer_age {
            if age < 25 || age > 65 {
                final_score = final_score.max(0.2);
                applied_rules.push("AGE_BASED_RISK".to_string());
            }
        }

        final_score = final_score.min(1.0);
        (final_score, applied_rules)
    }

    /// Determine risk level based on score
    fn determine_risk_level(&self, risk_score: f64) -> RiskLevel {
        if risk_score >= 0.9 {
            RiskLevel::Critical
        } else if risk_score >= 0.7 {
            RiskLevel::High
        } else if risk_score >= 0.4 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        }
    }

    /// Determine recommended action
    fn determine_recommended_action(&self, risk_level: RiskLevel) -> RecommendedAction {
        match risk_level {
            RiskLevel::Critical => RecommendedAction::Escalate,
            RiskLevel::High => RecommendedAction::Block,
            RiskLevel::Medium => RecommendedAction::Review,
            RiskLevel::Low => RecommendedAction::Approve,
        }
    }

    /// Fallback risk assessment when ML service fails
    fn fallback_risk_assessment(&self, request: &RiskAssessmentRequest) -> f64 {
        let mut risk_score = 0.0;

        // Amount-based risk
        if request.amount > 10000.0 {
            risk_score += 0.4;
        } else if request.amount > 5000.0 {
            risk_score += 0.2;
        }

        // Time-based risk
        let hour = request.timestamp.hour();
        if hour >= 22 || hour <= 6 {
            risk_score += 0.2;
        }

        // Weekend risk
        if request.timestamp.weekday().num_days_from_sunday() == 0 || 
           request.timestamp.weekday().num_days_from_sunday() == 6 {
            risk_score += 0.1;
        }

        // Age-based risk
        if let Some(age) = request.customer_age {
            if age < 25 || age > 65 {
                risk_score += 0.1;
            }
        }

        risk_score.min(1.0)
    }

    /// Generate model explanation
    async fn generate_explanation(&self, features: &HashMap<String, f64>, risk_score: f64) -> Result<Explanation> {
        // Get feature importance from ML service
        let feature_importance = self.ml_service.get_feature_importance().await?;
        
        // Get SHAP values
        let shap_values = self.ml_service.get_shap_values(features).await?;
        
        // Generate LIME explanation
        let lime_explanation = self.ml_service.get_lime_explanation(features).await?;
        
        // Create decision path
        let decision_path = self.create_decision_path(features, risk_score);

        Ok(Explanation {
            feature_importance,
            shap_values,
            lime_explanation,
            decision_path,
        })
    }

    /// Create decision path for explanation
    fn create_decision_path(&self, features: &HashMap<String, f64>, risk_score: f64) -> Vec<String> {
        let mut path = Vec::new();
        
        if let Some(amount) = features.get("amount") {
            if *amount > 10000.0 {
                path.push("High amount detected".to_string());
            }
        }
        
        if let Some(is_night) = features.get("is_night") {
            if *is_night > 0.0 {
                path.push("Transaction during night hours".to_string());
            }
        }
        
        if let Some(is_weekend) = features.get("is_weekend") {
            if *is_weekend > 0.0 {
                path.push("Weekend transaction".to_string());
            }
        }
        
        path.push(format!("Final risk score: {:.3}", risk_score));
        
        path
    }

    /// Get customer features from cache or database
    async fn get_customer_features(&self, customer_id: &str) -> Result<CustomerFeatures> {
        let cache_key = format!("customer:{}", customer_id);
        
        if let Some(cached) = self.cache.get(&cache_key).await? {
            return Ok(cached);
        }

        let features = self.database.get_customer_features(customer_id).await?;
        self.cache.set(&cache_key, &features, Duration::from_secs(300)).await?;
        
        Ok(features)
    }

    /// Get location features from cache or database
    async fn get_location_features(&self, location: &str) -> Result<LocationFeatures> {
        let cache_key = format!("location:{}", location);
        
        if let Some(cached) = self.cache.get(&cache_key).await? {
            return Ok(cached);
        }

        let features = self.database.get_location_features(location).await?;
        self.cache.set(&cache_key, &features, Duration::from_secs(600)).await?;
        
        Ok(features)
    }

    /// Publish risk event to Kafka
    async fn publish_risk_event(&self, response: &RiskAssessmentResponse) -> Result<()> {
        let event = RiskEvent {
            transaction_id: response.transaction_id,
            risk_score: response.risk_score,
            risk_level: response.risk_level.clone(),
            timestamp: response.timestamp,
        };

        self.kafka_producer.publish_risk_event(&event).await?;
        Ok(())
    }

    /// Start background tasks
    async fn start_background_tasks(&self) -> Result<()> {
        let engine = self.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                if let Err(e) = engine.update_metrics().await {
                    eprintln!("Failed to update metrics: {}", e);
                }
            }
        });

        Ok(())
    }

    /// Update engine metrics
    async fn update_metrics(&self) -> Result<()> {
        // Implementation to update metrics from database
        // This would typically involve querying the database for recent statistics
        Ok(())
    }

    /// Get engine metrics
    pub async fn get_metrics(&self) -> EngineMetrics {
        self.metrics.read().await.clone()
    }

    /// Get health status
    pub async fn health_check(&self) -> Result<HealthStatus> {
        let mut status = HealthStatus {
            status: "healthy".to_string(),
            timestamp: Utc::now(),
            services: HashMap::new(),
        };

        // Check ML service
        match self.ml_service.health_check().await {
            Ok(_) => status.services.insert("ml_service".to_string(), "healthy".to_string()),
            Err(_) => status.services.insert("ml_service".to_string(), "unhealthy".to_string()),
        }

        // Check cache
        match self.cache.health_check().await {
            Ok(_) => status.services.insert("cache".to_string(), "healthy".to_string()),
            Err(_) => status.services.insert("cache".to_string(), "unhealthy".to_string()),
        }

        // Check database
        match self.database.health_check().await {
            Ok(_) => status.services.insert("database".to_string(), "healthy".to_string()),
            Err(_) => status.services.insert("database".to_string(), "unhealthy".to_string()),
        }

        // Check Kafka
        match self.kafka_producer.health_check().await {
            Ok(_) => status.services.insert("kafka".to_string(), "healthy".to_string()),
            Err(_) => status.services.insert("kafka".to_string(), "unhealthy".to_string()),
        }

        // Determine overall status
        if status.services.values().any(|s| s == "unhealthy") {
            status.status = "degraded".to_string();
        }

        Ok(status)
    }
}

/// Customer features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerFeatures {
    pub transaction_count: f64,
    pub avg_amount: f64,
    pub risk_history: f64,
}

/// Location features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationFeatures {
    pub risk_score: f64,
    pub transaction_count: f64,
}

/// Risk event for Kafka
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskEvent {
    pub transaction_id: Uuid,
    pub risk_score: f64,
    pub risk_level: RiskLevel,
    pub timestamp: DateTime<Utc>,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: DateTime<Utc>,
    pub services: HashMap<String, String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_risk_assessment() {
        // Test implementation
        let request = RiskAssessmentRequest {
            transaction_id: Uuid::new_v4(),
            customer_id: "CUST123".to_string(),
            amount: 1000.0,
            timestamp: Utc::now(),
            location: Some("US".to_string()),
            merchant_id: Some("MERCHANT123".to_string()),
            customer_age: Some(30),
            payment_method: Some("CREDIT_CARD".to_string()),
            device_fingerprint: None,
            ip_address: None,
        };

        // Mock services and test
        // This would require setting up mock services
    }
}
