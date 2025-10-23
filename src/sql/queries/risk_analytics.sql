-- Advanced SQL Analytics for Financial Risk Assessment
-- Comprehensive data warehouse queries and analytics
-- 
-- Technologies: PostgreSQL, ClickHouse, Apache Spark SQL
-- Author: Nithin Yanna
-- Date: 2025

-- =====================================================
-- DATA WAREHOUSE SCHEMA
-- =====================================================

-- Create main tables
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255),
    phone VARCHAR(20),
    date_of_birth DATE,
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(50),
    country VARCHAR(50),
    postal_code VARCHAR(20),
    customer_segment VARCHAR(50),
    registration_date TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    risk_profile VARCHAR(20) DEFAULT 'MEDIUM',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    merchant_id VARCHAR(50),
    amount DECIMAL(15,2),
    currency VARCHAR(3) DEFAULT 'USD',
    transaction_type VARCHAR(50),
    payment_method VARCHAR(50),
    device_fingerprint VARCHAR(255),
    ip_address INET,
    location_country VARCHAR(50),
    location_region VARCHAR(50),
    location_city VARCHAR(100),
    latitude DECIMAL(10,8),
    longitude DECIMAL(11,8),
    transaction_timestamp TIMESTAMP,
    processing_timestamp TIMESTAMP,
    status VARCHAR(20) DEFAULT 'PENDING',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS risk_assessments (
    assessment_id VARCHAR(50) PRIMARY KEY,
    transaction_id VARCHAR(50) REFERENCES transactions(transaction_id),
    customer_id VARCHAR(50),
    risk_score DECIMAL(5,4),
    risk_level VARCHAR(20),
    confidence_score DECIMAL(5,4),
    recommended_action VARCHAR(20),
    processing_time_ms INTEGER,
    model_version VARCHAR(50),
    features JSONB,
    explanation JSONB,
    business_rules_applied TEXT[],
    assessment_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS merchants (
    merchant_id VARCHAR(50) PRIMARY KEY,
    merchant_name VARCHAR(255),
    merchant_category VARCHAR(100),
    merchant_country VARCHAR(50),
    merchant_region VARCHAR(50),
    risk_score DECIMAL(5,4),
    is_high_risk BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_performance (
    model_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    accuracy DECIMAL(5,4),
    precision_score DECIMAL(5,4),
    recall_score DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    auc_score DECIMAL(5,4),
    confusion_matrix JSONB,
    feature_importance JSONB,
    drift_score DECIMAL(5,4),
    training_samples INTEGER,
    validation_samples INTEGER,
    training_duration_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS system_metrics (
    metric_id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_value DECIMAL(15,4),
    metric_unit VARCHAR(20),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    service_name VARCHAR(100),
    environment VARCHAR(50)
);

-- =====================================================
-- ADVANCED ANALYTICS QUERIES
-- =====================================================

-- 1. Real-time Risk Dashboard Metrics
WITH risk_summary AS (
    SELECT 
        DATE_TRUNC('hour', assessment_timestamp) as hour,
        COUNT(*) as total_assessments,
        COUNT(CASE WHEN risk_level = 'HIGH' THEN 1 END) as high_risk_count,
        COUNT(CASE WHEN risk_level = 'MEDIUM' THEN 1 END) as medium_risk_count,
        COUNT(CASE WHEN risk_level = 'LOW' THEN 1 END) as low_risk_count,
        AVG(risk_score) as avg_risk_score,
        AVG(processing_time_ms) as avg_processing_time,
        COUNT(CASE WHEN recommended_action = 'BLOCK' THEN 1 END) as blocked_transactions
    FROM risk_assessments 
    WHERE assessment_timestamp >= NOW() - INTERVAL '24 hours'
    GROUP BY DATE_TRUNC('hour', assessment_timestamp)
)
SELECT 
    hour,
    total_assessments,
    high_risk_count,
    medium_risk_count,
    low_risk_count,
    ROUND(avg_risk_score, 4) as avg_risk_score,
    ROUND(avg_processing_time, 2) as avg_processing_time_ms,
    blocked_transactions,
    ROUND((high_risk_count::DECIMAL / total_assessments) * 100, 2) as high_risk_percentage,
    ROUND((blocked_transactions::DECIMAL / total_assessments) * 100, 2) as block_rate
FROM risk_summary
ORDER BY hour DESC;

-- 2. Customer Risk Profile Analysis
WITH customer_risk_profiles AS (
    SELECT 
        c.customer_id,
        c.customer_segment,
        c.registration_date,
        COUNT(ra.assessment_id) as total_assessments,
        AVG(ra.risk_score) as avg_risk_score,
        MAX(ra.risk_score) as max_risk_score,
        COUNT(CASE WHEN ra.risk_level = 'HIGH' THEN 1 END) as high_risk_assessments,
        COUNT(CASE WHEN ra.recommended_action = 'BLOCK' THEN 1 END) as blocked_count,
        SUM(t.amount) as total_transaction_amount,
        AVG(t.amount) as avg_transaction_amount,
        COUNT(DISTINCT t.merchant_id) as unique_merchants,
        COUNT(DISTINCT t.location_country) as unique_countries
    FROM customers c
    LEFT JOIN risk_assessments ra ON c.customer_id = ra.customer_id
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    WHERE ra.assessment_timestamp >= NOW() - INTERVAL '30 days'
    GROUP BY c.customer_id, c.customer_segment, c.registration_date
)
SELECT 
    customer_segment,
    COUNT(*) as customer_count,
    ROUND(AVG(avg_risk_score), 4) as segment_avg_risk_score,
    ROUND(AVG(total_assessments), 2) as avg_assessments_per_customer,
    ROUND(AVG(total_transaction_amount), 2) as avg_total_amount,
    ROUND(AVG(unique_merchants), 2) as avg_unique_merchants,
    COUNT(CASE WHEN avg_risk_score > 0.7 THEN 1 END) as high_risk_customers,
    ROUND(COUNT(CASE WHEN avg_risk_score > 0.7 THEN 1 END)::DECIMAL / COUNT(*) * 100, 2) as high_risk_percentage
FROM customer_risk_profiles
GROUP BY customer_segment
ORDER BY segment_avg_risk_score DESC;

-- 3. Geographic Risk Analysis
WITH geographic_risk AS (
    SELECT 
        t.location_country,
        t.location_region,
        t.location_city,
        COUNT(*) as transaction_count,
        AVG(ra.risk_score) as avg_risk_score,
        COUNT(CASE WHEN ra.risk_level = 'HIGH' THEN 1 END) as high_risk_count,
        SUM(t.amount) as total_amount,
        AVG(t.amount) as avg_amount,
        COUNT(DISTINCT t.customer_id) as unique_customers,
        COUNT(DISTINCT t.merchant_id) as unique_merchants
    FROM transactions t
    JOIN risk_assessments ra ON t.transaction_id = ra.transaction_id
    WHERE t.transaction_timestamp >= NOW() - INTERVAL '7 days'
    GROUP BY t.location_country, t.location_region, t.location_city
    HAVING COUNT(*) >= 10  -- Only locations with significant activity
)
SELECT 
    location_country,
    location_region,
    location_city,
    transaction_count,
    ROUND(avg_risk_score, 4) as avg_risk_score,
    high_risk_count,
    ROUND((high_risk_count::DECIMAL / transaction_count) * 100, 2) as high_risk_percentage,
    ROUND(total_amount, 2) as total_amount,
    ROUND(avg_amount, 2) as avg_amount,
    unique_customers,
    unique_merchants,
    ROUND(avg_amount * unique_customers, 2) as risk_weighted_score
FROM geographic_risk
ORDER BY avg_risk_score DESC, transaction_count DESC
LIMIT 50;

-- 4. Merchant Risk Analysis
WITH merchant_risk AS (
    SELECT 
        m.merchant_id,
        m.merchant_name,
        m.merchant_category,
        m.merchant_country,
        COUNT(t.transaction_id) as transaction_count,
        AVG(ra.risk_score) as avg_risk_score,
        COUNT(CASE WHEN ra.risk_level = 'HIGH' THEN 1 END) as high_risk_count,
        SUM(t.amount) as total_amount,
        AVG(t.amount) as avg_amount,
        COUNT(DISTINCT t.customer_id) as unique_customers,
        COUNT(CASE WHEN ra.recommended_action = 'BLOCK' THEN 1 END) as blocked_transactions
    FROM merchants m
    JOIN transactions t ON m.merchant_id = t.merchant_id
    JOIN risk_assessments ra ON t.transaction_id = ra.transaction_id
    WHERE t.transaction_timestamp >= NOW() - INTERVAL '30 days'
    GROUP BY m.merchant_id, m.merchant_name, m.merchant_category, m.merchant_country
    HAVING COUNT(t.transaction_id) >= 5
)
SELECT 
    merchant_id,
    merchant_name,
    merchant_category,
    merchant_country,
    transaction_count,
    ROUND(avg_risk_score, 4) as avg_risk_score,
    high_risk_count,
    ROUND((high_risk_count::DECIMAL / transaction_count) * 100, 2) as high_risk_percentage,
    ROUND(total_amount, 2) as total_amount,
    ROUND(avg_amount, 2) as avg_amount,
    unique_customers,
    blocked_transactions,
    ROUND((blocked_transactions::DECIMAL / transaction_count) * 100, 2) as block_rate,
    CASE 
        WHEN avg_risk_score > 0.8 THEN 'CRITICAL'
        WHEN avg_risk_score > 0.6 THEN 'HIGH'
        WHEN avg_risk_score > 0.4 THEN 'MEDIUM'
        ELSE 'LOW'
    END as merchant_risk_level
FROM merchant_risk
ORDER BY avg_risk_score DESC, transaction_count DESC;

-- 5. Time-based Risk Patterns
WITH hourly_patterns AS (
    SELECT 
        EXTRACT(HOUR FROM t.transaction_timestamp) as hour_of_day,
        EXTRACT(DOW FROM t.transaction_timestamp) as day_of_week,
        COUNT(*) as transaction_count,
        AVG(ra.risk_score) as avg_risk_score,
        COUNT(CASE WHEN ra.risk_level = 'HIGH' THEN 1 END) as high_risk_count,
        SUM(t.amount) as total_amount,
        AVG(t.amount) as avg_amount
    FROM transactions t
    JOIN risk_assessments ra ON t.transaction_id = ra.transaction_id
    WHERE t.transaction_timestamp >= NOW() - INTERVAL '30 days'
    GROUP BY EXTRACT(HOUR FROM t.transaction_timestamp), EXTRACT(DOW FROM t.transaction_timestamp)
)
SELECT 
    hour_of_day,
    CASE day_of_week
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END as day_name,
    transaction_count,
    ROUND(avg_risk_score, 4) as avg_risk_score,
    high_risk_count,
    ROUND((high_risk_count::DECIMAL / transaction_count) * 100, 2) as high_risk_percentage,
    ROUND(total_amount, 2) as total_amount,
    ROUND(avg_amount, 2) as avg_amount
FROM hourly_patterns
ORDER BY day_of_week, hour_of_day;

-- 6. Model Performance Analysis
WITH model_performance_metrics AS (
    SELECT 
        mp.model_name,
        mp.model_version,
        mp.accuracy,
        mp.precision_score,
        mp.recall_score,
        mp.f1_score,
        mp.auc_score,
        mp.drift_score,
        mp.training_samples,
        mp.validation_samples,
        mp.training_duration_seconds,
        mp.created_at,
        -- Calculate performance trend
        LAG(mp.accuracy) OVER (PARTITION BY mp.model_name ORDER BY mp.created_at) as prev_accuracy,
        LAG(mp.f1_score) OVER (PARTITION BY mp.model_name ORDER BY mp.created_at) as prev_f1_score
    FROM model_performance mp
    WHERE mp.created_at >= NOW() - INTERVAL '30 days'
)
SELECT 
    model_name,
    model_version,
    ROUND(accuracy, 4) as accuracy,
    ROUND(precision_score, 4) as precision,
    ROUND(recall_score, 4) as recall,
    ROUND(f1_score, 4) as f1_score,
    ROUND(auc_score, 4) as auc,
    ROUND(drift_score, 4) as drift_score,
    training_samples,
    validation_samples,
    training_duration_seconds,
    ROUND(accuracy - COALESCE(prev_accuracy, accuracy), 4) as accuracy_change,
    ROUND(f1_score - COALESCE(prev_f1_score, f1_score), 4) as f1_change,
    CASE 
        WHEN drift_score > 0.1 THEN 'HIGH_DRIFT'
        WHEN drift_score > 0.05 THEN 'MEDIUM_DRIFT'
        ELSE 'LOW_DRIFT'
    END as drift_status,
    created_at
FROM model_performance_metrics
ORDER BY created_at DESC;

-- 7. Feature Importance Analysis
WITH feature_importance AS (
    SELECT 
        mp.model_name,
        mp.model_version,
        mp.created_at,
        jsonb_each_text(mp.feature_importance) as feature_data
    FROM model_performance mp
    WHERE mp.created_at >= NOW() - INTERVAL '7 days'
),
feature_ranks AS (
    SELECT 
        model_name,
        model_version,
        created_at,
        (feature_data).key as feature_name,
        (feature_data).value::DECIMAL as importance_score,
        ROW_NUMBER() OVER (PARTITION BY model_name, model_version ORDER BY (feature_data).value::DECIMAL DESC) as feature_rank
    FROM feature_importance
)
SELECT 
    model_name,
    model_version,
    feature_name,
    ROUND(importance_score, 6) as importance_score,
    feature_rank,
    created_at
FROM feature_ranks
WHERE feature_rank <= 20  -- Top 20 features
ORDER BY model_name, model_version, feature_rank;

-- 8. System Performance Metrics
WITH system_metrics_hourly AS (
    SELECT 
        DATE_TRUNC('hour', timestamp) as hour,
        metric_name,
        AVG(metric_value) as avg_value,
        MAX(metric_value) as max_value,
        MIN(metric_value) as min_value,
        STDDEV(metric_value) as stddev_value
    FROM system_metrics
    WHERE timestamp >= NOW() - INTERVAL '24 hours'
    GROUP BY DATE_TRUNC('hour', timestamp), metric_name
)
SELECT 
    hour,
    metric_name,
    ROUND(avg_value, 4) as avg_value,
    ROUND(max_value, 4) as max_value,
    ROUND(min_value, 4) as min_value,
    ROUND(stddev_value, 4) as stddev_value,
    CASE 
        WHEN metric_name = 'cpu_usage' AND avg_value > 80 THEN 'HIGH_CPU'
        WHEN metric_name = 'memory_usage' AND avg_value > 80 THEN 'HIGH_MEMORY'
        WHEN metric_name = 'error_rate' AND avg_value > 5 THEN 'HIGH_ERROR_RATE'
        ELSE 'NORMAL'
    END as alert_status
FROM system_metrics_hourly
ORDER BY hour DESC, metric_name;

-- 9. Risk Score Distribution Analysis
WITH risk_distribution AS (
    SELECT 
        CASE 
            WHEN risk_score < 0.2 THEN '0.0-0.2'
            WHEN risk_score < 0.4 THEN '0.2-0.4'
            WHEN risk_score < 0.6 THEN '0.4-0.6'
            WHEN risk_score < 0.8 THEN '0.6-0.8'
            ELSE '0.8-1.0'
        END as risk_score_bucket,
        COUNT(*) as assessment_count,
        AVG(processing_time_ms) as avg_processing_time,
        COUNT(CASE WHEN recommended_action = 'BLOCK' THEN 1 END) as blocked_count
    FROM risk_assessments
    WHERE assessment_timestamp >= NOW() - INTERVAL '7 days'
    GROUP BY 
        CASE 
            WHEN risk_score < 0.2 THEN '0.0-0.2'
            WHEN risk_score < 0.4 THEN '0.2-0.4'
            WHEN risk_score < 0.6 THEN '0.4-0.6'
            WHEN risk_score < 0.8 THEN '0.6-0.8'
            ELSE '0.8-1.0'
        END
)
SELECT 
    risk_score_bucket,
    assessment_count,
    ROUND((assessment_count::DECIMAL / SUM(assessment_count) OVER ()) * 100, 2) as percentage,
    ROUND(avg_processing_time, 2) as avg_processing_time_ms,
    blocked_count,
    ROUND((blocked_count::DECIMAL / assessment_count) * 100, 2) as block_rate
FROM risk_distribution
ORDER BY risk_score_bucket;

-- 10. Advanced Customer Segmentation
WITH customer_segments AS (
    SELECT 
        c.customer_id,
        c.customer_segment,
        COUNT(ra.assessment_id) as total_assessments,
        AVG(ra.risk_score) as avg_risk_score,
        STDDEV(ra.risk_score) as risk_score_stddev,
        COUNT(CASE WHEN ra.risk_level = 'HIGH' THEN 1 END) as high_risk_count,
        SUM(t.amount) as total_amount,
        AVG(t.amount) as avg_amount,
        COUNT(DISTINCT t.merchant_id) as unique_merchants,
        COUNT(DISTINCT t.location_country) as unique_countries,
        MAX(t.transaction_timestamp) as last_transaction,
        MIN(t.transaction_timestamp) as first_transaction,
        EXTRACT(DAYS FROM MAX(t.transaction_timestamp) - MIN(t.transaction_timestamp)) as customer_lifespan_days
    FROM customers c
    LEFT JOIN risk_assessments ra ON c.customer_id = ra.customer_id
    LEFT JOIN transactions t ON c.customer_id = t.customer_id
    WHERE ra.assessment_timestamp >= NOW() - INTERVAL '90 days'
    GROUP BY c.customer_id, c.customer_segment
    HAVING COUNT(ra.assessment_id) >= 1
),
segmented_customers AS (
    SELECT 
        customer_id,
        customer_segment,
        total_assessments,
        avg_risk_score,
        risk_score_stddev,
        high_risk_count,
        total_amount,
        avg_amount,
        unique_merchants,
        unique_countries,
        last_transaction,
        first_transaction,
        customer_lifespan_days,
        -- Advanced segmentation logic
        CASE 
            WHEN avg_risk_score > 0.7 AND high_risk_count > 2 THEN 'HIGH_RISK_FREQUENT'
            WHEN avg_risk_score > 0.7 AND high_risk_count <= 2 THEN 'HIGH_RISK_OCCASIONAL'
            WHEN avg_risk_score BETWEEN 0.4 AND 0.7 AND total_amount > 10000 THEN 'MEDIUM_RISK_HIGH_VALUE'
            WHEN avg_risk_score BETWEEN 0.4 AND 0.7 AND total_amount <= 10000 THEN 'MEDIUM_RISK_STANDARD'
            WHEN avg_risk_score < 0.4 AND total_amount > 50000 THEN 'LOW_RISK_PREMIUM'
            WHEN avg_risk_score < 0.4 AND total_amount BETWEEN 10000 AND 50000 THEN 'LOW_RISK_STANDARD'
            ELSE 'LOW_RISK_BASIC'
        END as advanced_segment,
        -- Risk volatility
        CASE 
            WHEN risk_score_stddev > 0.3 THEN 'HIGH_VOLATILITY'
            WHEN risk_score_stddev BETWEEN 0.1 AND 0.3 THEN 'MEDIUM_VOLATILITY'
            ELSE 'LOW_VOLATILITY'
        END as risk_volatility
    FROM customer_segments
)
SELECT 
    advanced_segment,
    risk_volatility,
    COUNT(*) as customer_count,
    ROUND(AVG(avg_risk_score), 4) as segment_avg_risk_score,
    ROUND(AVG(total_amount), 2) as segment_avg_total_amount,
    ROUND(AVG(unique_merchants), 2) as segment_avg_merchants,
    ROUND(AVG(unique_countries), 2) as segment_avg_countries,
    ROUND(AVG(customer_lifespan_days), 2) as segment_avg_lifespan_days,
    COUNT(CASE WHEN high_risk_count > 0 THEN 1 END) as customers_with_high_risk,
    ROUND(COUNT(CASE WHEN high_risk_count > 0 THEN 1 END)::DECIMAL / COUNT(*) * 100, 2) as high_risk_percentage
FROM segmented_customers
GROUP BY advanced_segment, risk_volatility
ORDER BY segment_avg_risk_score DESC, customer_count DESC;

-- =====================================================
-- PERFORMANCE OPTIMIZATION QUERIES
-- =====================================================

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_risk_assessments_timestamp ON risk_assessments(assessment_timestamp);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_customer_id ON risk_assessments(customer_id);
CREATE INDEX IF NOT EXISTS idx_risk_assessments_risk_level ON risk_assessments(risk_level);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(transaction_timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant_id ON transactions(merchant_id);
CREATE INDEX IF NOT EXISTS idx_transactions_location ON transactions(location_country, location_region);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);

-- Create materialized views for frequently accessed data
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_daily_risk_summary AS
SELECT 
    DATE(assessment_timestamp) as assessment_date,
    COUNT(*) as total_assessments,
    COUNT(CASE WHEN risk_level = 'HIGH' THEN 1 END) as high_risk_count,
    COUNT(CASE WHEN risk_level = 'MEDIUM' THEN 1 END) as medium_risk_count,
    COUNT(CASE WHEN risk_level = 'LOW' THEN 1 END) as low_risk_count,
    AVG(risk_score) as avg_risk_score,
    AVG(processing_time_ms) as avg_processing_time,
    COUNT(CASE WHEN recommended_action = 'BLOCK' THEN 1 END) as blocked_count
FROM risk_assessments
GROUP BY DATE(assessment_timestamp);

CREATE UNIQUE INDEX ON mv_daily_risk_summary(assessment_date);

-- Refresh materialized view (run this periodically)
-- REFRESH MATERIALIZED VIEW mv_daily_risk_summary;

-- =====================================================
-- DATA QUALITY AND MONITORING QUERIES
-- =====================================================

-- Data quality check for risk assessments
SELECT 
    'risk_assessments' as table_name,
    COUNT(*) as total_rows,
    COUNT(CASE WHEN risk_score IS NULL THEN 1 END) as null_risk_scores,
    COUNT(CASE WHEN risk_level IS NULL THEN 1 END) as null_risk_levels,
    COUNT(CASE WHEN customer_id IS NULL THEN 1 END) as null_customer_ids,
    COUNT(CASE WHEN transaction_id IS NULL THEN 1 END) as null_transaction_ids,
    COUNT(CASE WHEN risk_score < 0 OR risk_score > 1 THEN 1 END) as invalid_risk_scores,
    MIN(assessment_timestamp) as earliest_timestamp,
    MAX(assessment_timestamp) as latest_timestamp
FROM risk_assessments;

-- Data quality check for transactions
SELECT 
    'transactions' as table_name,
    COUNT(*) as total_rows,
    COUNT(CASE WHEN amount IS NULL THEN 1 END) as null_amounts,
    COUNT(CASE WHEN amount <= 0 THEN 1 END) as invalid_amounts,
    COUNT(CASE WHEN customer_id IS NULL THEN 1 END) as null_customer_ids,
    COUNT(CASE WHEN transaction_timestamp IS NULL THEN 1 END) as null_timestamps,
    MIN(transaction_timestamp) as earliest_timestamp,
    MAX(transaction_timestamp) as latest_timestamp
FROM transactions;

-- System health monitoring
SELECT 
    metric_name,
    COUNT(*) as data_points,
    ROUND(AVG(metric_value), 4) as avg_value,
    ROUND(MAX(metric_value), 4) as max_value,
    ROUND(MIN(metric_value), 4) as min_value,
    ROUND(STDDEV(metric_value), 4) as stddev_value,
    MAX(timestamp) as latest_reading
FROM system_metrics
WHERE timestamp >= NOW() - INTERVAL '1 hour'
GROUP BY metric_name
ORDER BY metric_name;
