"""
Advanced Machine Learning Models for Financial Risk Assessment
Multi-language, multi-framework implementation

Technologies: PyTorch, TensorFlow, XGBoost, LightGBM, CatBoost, Optuna
Author: Nithin Yanna
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
from optuna.integration import XGBoostPruningCallback
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerRiskModel(nn.Module):
    """
    Advanced Transformer-based model for financial risk assessment
    with multi-head attention and positional encoding
    """
    
    def __init__(self, input_dim: int, d_model: int = 512, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1, max_seq_len: int = 1000):
        super(TransformerRiskModel, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Attention weights for interpretability
        self.attention_weights = None
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding for transformer"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        output = self.classifier(x)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention weights for interpretability"""
        self.eval()
        with torch.no_grad():
            batch_size, seq_len, _ = x.shape
            x = self.input_projection(x)
            x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
            
            # Get attention weights from transformer
            for layer in self.transformer.layers:
                x, attn_weights = layer.self_attn(x, x, x, need_weights=True)
                self.attention_weights = attn_weights
            
            return self.attention_weights

class AdvancedTensorFlowModel(keras.Model):
    """
    Advanced TensorFlow model with custom layers and advanced architectures
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128], 
                 dropout_rate: float = 0.3, use_attention: bool = True):
        super(AdvancedTensorFlowModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # Build layers
        self._build_layers()
        
    def _build_layers(self):
        """Build the model layers"""
        self.dense_layers = []
        self.dropout_layers = []
        self.batch_norm_layers = []
        
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.dense_layers.append(keras.layers.Dense(hidden_dim, activation='relu'))
            self.batch_norm_layers.append(keras.layers.BatchNormalization())
            self.dropout_layers.append(keras.layers.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        
        # Attention mechanism
        if self.use_attention:
            self.attention = keras.layers.MultiHeadAttention(
                num_heads=8, key_dim=prev_dim // 8
            )
            self.layer_norm = keras.layers.LayerNormalization()
        
        # Output layer
        self.output_layer = keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=None):
        x = inputs
        
        # Dense layers with batch norm and dropout
        for dense, bn, dropout in zip(self.dense_layers, self.batch_norm_layers, self.dropout_layers):
            x = dense(x)
            x = bn(x, training=training)
            x = dropout(x, training=training)
        
        # Attention mechanism
        if self.use_attention:
            # Reshape for attention (batch_size, 1, features)
            x_reshaped = tf.expand_dims(x, axis=1)
            attn_output = self.attention(x_reshaped, x_reshaped)
            attn_output = tf.squeeze(attn_output, axis=1)
            x = self.layer_norm(x + attn_output)
        
        # Output
        return self.output_layer(x)

class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization using Optuna
    """
    
    def __init__(self, n_trials: int = 100, timeout: int = 3600):
        self.n_trials = n_trials
        self.timeout = timeout
        self.study = None
        
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray, 
                         cv_folds: int = 5) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
            return scores.mean()
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        return self.study.best_params
    
    def optimize_lightgbm(self, X: np.ndarray, y: np.ndarray, 
                          cv_folds: int = 5) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42,
                'verbose': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            scores = cross_val_score(model, X, y, cv=cv_folds, scoring='roc_auc')
            return scores.mean()
        
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        return self.study.best_params

class AdvancedEnsembleModel:
    """
    Advanced ensemble model with multiple algorithms and sophisticated weighting
    """
    
    def __init__(self, use_mlflow: bool = True):
        self.models = {}
        self.weights = {}
        self.feature_importance = {}
        self.explainers = {}
        self.use_mlflow = use_mlflow
        
        if self.use_mlflow:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow.set_experiment("financial_risk_assessment")
    
    def _initialize_models(self, X: np.ndarray, y: np.ndarray):
        """Initialize all model components"""
        
        # XGBoost with optimization
        optimizer = HyperparameterOptimizer(n_trials=50)
        xgb_params = optimizer.optimize_xgboost(X, y)
        self.models['xgboost'] = xgb.XGBClassifier(**xgb_params)
        
        # LightGBM with optimization
        lgb_params = optimizer.optimize_lightgbm(X, y)
        self.models['lightgbm'] = lgb.LGBMClassifier(**lgb_params)
        
        # CatBoost
        self.models['catboost'] = cb.CatBoostClassifier(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            random_seed=42,
            verbose=False
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42
        )
        
        # SVM
        self.models['svm'] = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
        
        # Neural Network
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )
        
        # PyTorch Transformer
        self.models['transformer'] = TransformerRiskModel(
            input_dim=X.shape[1],
            d_model=256,
            nhead=8,
            num_layers=4
        )
        
        # TensorFlow model
        self.models['tensorflow'] = AdvancedTensorFlowModel(
            input_dim=X.shape[1],
            hidden_dims=[512, 256, 128],
            dropout_rate=0.3
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: List[str] = None) -> 'AdvancedEnsembleModel':
        """Train the ensemble model"""
        
        if self.use_mlflow:
            with mlflow.start_run():
                mlflow.log_param("n_samples", X.shape[0])
                mlflow.log_param("n_features", X.shape[1])
        
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Initialize models
        self._initialize_models(X, y)
        
        # Train each model
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            if name == 'transformer':
                # PyTorch model training
                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.FloatTensor(y).unsqueeze(1)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.BCELoss()
                
                model.train()
                for epoch in range(100):
                    optimizer.zero_grad()
                    outputs = model(X_tensor)
                    loss = criterion(outputs, y_tensor)
                    loss.backward()
                    optimizer.step()
                    
                    if epoch % 20 == 0:
                        logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            elif name == 'tensorflow':
                # TensorFlow model training
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                model.fit(X, y, epochs=100, batch_size=32, verbose=0)
            
            else:
                # Scikit-learn models
                model.fit(X, y)
        
        # Optimize ensemble weights
        self._optimize_weights(X, y)
        
        # Train explainers
        self._train_explainers(X, y)
        
        logger.info("Ensemble model training completed")
        return self
    
    def _optimize_weights(self, X: np.ndarray, y: np.ndarray):
        """Optimize ensemble weights using validation performance"""
        
        def weight_objective(trial):
            weights = []
            for i in range(len(self.models)):
                weights.append(trial.suggest_float(f'weight_{i}', 0, 1))
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # Get predictions
            predictions = self._get_ensemble_predictions(X, weights)
            
            # Calculate score
            score = roc_auc_score(y, predictions)
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(weight_objective, n_trials=100)
        
        # Set optimized weights
        best_weights = []
        for i in range(len(self.models)):
            best_weights.append(study.best_params[f'weight_{i}'])
        
        self.weights = dict(zip(self.models.keys(), best_weights))
        logger.info(f"Optimized weights: {self.weights}")
    
    def _get_ensemble_predictions(self, X: np.ndarray, weights: List[float]) -> np.ndarray:
        """Get ensemble predictions with given weights"""
        predictions = []
        
        for name, model in self.models.items():
            if name == 'transformer':
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    pred = model(X_tensor).numpy().flatten()
            elif name == 'tensorflow':
                pred = model.predict(X, verbose=0).flatten()
            else:
                pred = model.predict_proba(X)[:, 1]
            
            predictions.append(pred)
        
        # Weighted ensemble
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def _train_explainers(self, X: np.ndarray, y: np.ndarray):
        """Train model explainers"""
        logger.info("Training model explainers...")
        
        # SHAP explainer for tree-based models
        if 'xgboost' in self.models:
            self.explainers['shap'] = shap.TreeExplainer(self.models['xgboost'])
        
        # LIME explainer
        self.explainers['lime'] = lime.lime_tabular.LimeTabularExplainer(
            X, feature_names=self.feature_names, mode='classification'
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        predictions = []
        weights = list(self.weights.values())
        
        for name, model in self.models.items():
            if name == 'transformer':
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    pred = model(X_tensor).numpy().flatten()
            elif name == 'tensorflow':
                pred = model.predict(X, verbose=0).flatten()
            else:
                pred = model.predict_proba(X)[:, 1]
            
            predictions.append(pred)
        
        # Weighted ensemble
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble probabilities"""
        predictions = []
        weights = list(self.weights.values())
        
        for name, model in self.models.items():
            if name == 'transformer':
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X)
                    pred = model(X_tensor).numpy().flatten()
            elif name == 'tensorflow':
                pred = model.predict(X, verbose=0).flatten()
            else:
                pred = model.predict_proba(X)[:, 1]
            
            predictions.append(pred)
        
        # Weighted ensemble
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get ensemble feature importance"""
        importance = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                model_importance = model.feature_importances_
                weight = self.weights[name]
                
                for i, feature in enumerate(self.feature_names):
                    if feature not in importance:
                        importance[feature] = 0
                    importance[feature] += weight * model_importance[i]
        
        return importance
    
    def explain_prediction(self, X: np.ndarray, instance_idx: int = 0) -> Dict:
        """Explain individual predictions"""
        explanations = {}
        
        # SHAP explanation
        if 'shap' in self.explainers:
            shap_values = self.explainers['shap'].shap_values(X[instance_idx:instance_idx+1])
            explanations['shap'] = {
                'values': shap_values[0].tolist(),
                'features': self.feature_names
            }
        
        # LIME explanation
        if 'lime' in self.explainers:
            lime_exp = self.explainers['lime'].explain_instance(
                X[instance_idx], self.predict_proba, num_features=10
            )
            explanations['lime'] = {
                'features': [x[0] for x in lime_exp.as_list()],
                'scores': [x[1] for x in lime_exp.as_list()]
            }
        
        return explanations
    
    def save_models(self, path: str):
        """Save all models"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            if name in ['transformer', 'tensorflow']:
                # Save PyTorch/TensorFlow models
                torch.save(model.state_dict(), f"{path}/{name}_model.pth")
            else:
                # Save scikit-learn models
                joblib.dump(model, f"{path}/{name}_model.pkl")
        
        # Save ensemble metadata
        joblib.dump({
            'weights': self.weights,
            'feature_names': self.feature_names
        }, f"{path}/ensemble_metadata.pkl")

# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 100)
    y = np.random.randint(0, 2, 1000)
    feature_names = [f"feature_{i}" for i in range(100)]
    
    # Train ensemble model
    model = AdvancedEnsembleModel(use_mlflow=True)
    model.fit(X, y, feature_names)
    
    # Make predictions
    predictions = model.predict(X[:10])
    probabilities = model.predict_proba(X[:10])
    
    # Get feature importance
    importance = model.get_feature_importance()
    
    # Explain predictions
    explanations = model.explain_prediction(X, instance_idx=0)
    
    print("Advanced ensemble model training completed!")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Top 5 important features: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
    
    # Save models
    model.save_models("models/")
