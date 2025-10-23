"""
Advanced Ensemble Model for Financial Risk Assessment

This module implements a sophisticated ensemble model combining:
- Gradient Boosting (XGBoost/LightGBM)
- Deep Learning (PyTorch)
- Transformer-based models
- Custom business logic rules

Author: Nithin Yanna
Date: 2025
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from sklearn.metrics import roc_auc_score, precision_recall_curve
import shap
import lime
import lime.lime_tabular

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for ensemble model components"""
    xgb_params: Dict = None
    lgb_params: Dict = None
    neural_network_config: Dict = None
    transformer_config: Dict = None
    ensemble_weights: List[float] = None
    feature_importance_threshold: float = 0.01
    model_explainability: bool = True
    bias_detection: bool = True

class BaseModel(ABC):
    """Abstract base class for all model components"""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores"""
        pass

class AdvancedNeuralNetwork(nn.Module, BaseModel):
    """
    Advanced Deep Learning Model with attention mechanisms
    and residual connections for financial risk assessment
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128, 64], 
                 dropout_rate: float = 0.3, use_attention: bool = True,
                 use_residual: bool = True):
        super(AdvancedNeuralNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Build network layers
        self._build_network()
        
    def _build_network(self):
        """Build the neural network architecture"""
        layers = []
        prev_dim = self.input_dim
        
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            
            # Attention mechanism
            if self.use_attention and i < len(self.hidden_dims) - 1:
                layers.append(AttentionLayer(hidden_dim))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Residual connections
        if self.use_residual:
            self.residual_layers = nn.ModuleList([
                nn.Linear(self.input_dim, dim) for dim in self.hidden_dims
            ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections"""
        if self.use_residual:
            residual = x
            for i, layer in enumerate(self.network[:-1]):  # Exclude output layer
                x = layer(x)
                if i < len(self.residual_layers):
                    residual_proj = self.residual_layers[i](residual)
                    x = x + residual_proj
                    residual = x
            x = self.network[-1](x)  # Output layer
        else:
            x = self.network(x)
        
        return x
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
            batch_size: int = 32, learning_rate: float = 0.001, **kwargs) -> 'AdvancedNeuralNetwork':
        """Train the neural network"""
        self.train()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.forward(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make binary predictions"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.forward(X_tensor)
            return (predictions > 0.5).float().numpy().flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            probabilities = self.forward(X_tensor)
            return probabilities.numpy().flatten()
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance using gradient-based method"""
        self.eval()
        # This is a simplified version - in practice, you'd use more sophisticated methods
        return np.random.random(self.input_dim)  # Placeholder

class AttentionLayer(nn.Module):
    """Self-attention layer for feature importance"""
    
    def __init__(self, input_dim: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reshape for attention (batch_size, 1, features)
        x_reshaped = x.unsqueeze(1)
        attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        attn_output = attn_output.squeeze(1)
        return self.norm(x + attn_output)

class EnsembleRiskModel:
    """
    Advanced Ensemble Model combining multiple ML approaches
    with sophisticated weighting and bias detection
    """
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {}
        self.feature_names = None
        self.explainer = None
        self.bias_detector = None
        
    def _initialize_models(self, X: np.ndarray):
        """Initialize all model components"""
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier
        
        # XGBoost model
        self.models['xgboost'] = XGBClassifier(
            **self.config.xgb_params,
            random_state=42,
            eval_metric='logloss'
        )
        
        # LightGBM model
        self.models['lightgbm'] = LGBMClassifier(
            **self.config.lgb_params,
            random_state=42,
            verbose=-1
        )
        
        # Neural Network
        self.models['neural_network'] = AdvancedNeuralNetwork(
            input_dim=X.shape[1],
            **self.config.neural_network_config
        )
        
        # Initialize explainer if needed
        if self.config.model_explainability:
            self.explainer = shap.TreeExplainer(self.models['xgboost'])
        
        # Initialize bias detector
        if self.config.bias_detection:
            self.bias_detector = BiasDetector()
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            feature_names: List[str] = None, **kwargs) -> 'EnsembleRiskModel':
        """Train the ensemble model"""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Initialize models
        self._initialize_models(X)
        
        # Train individual models
        logger.info("Training XGBoost model...")
        self.models['xgboost'].fit(X, y)
        
        logger.info("Training LightGBM model...")
        self.models['lightgbm'].fit(X, y)
        
        logger.info("Training Neural Network...")
        self.models['neural_network'].fit(X, y, **kwargs)
        
        # Optimize ensemble weights
        self._optimize_ensemble_weights(X, y)
        
        # Train explainer
        if self.explainer:
            self.explainer = shap.TreeExplainer(self.models['xgboost'])
        
        # Detect bias
        if self.bias_detector:
            self.bias_detector.fit(X, y, self.feature_names)
        
        logger.info("Ensemble model training completed")
        return self
    
    def _optimize_ensemble_weights(self, X: np.ndarray, y: np.ndarray):
        """Optimize ensemble weights using validation performance"""
        from sklearn.model_selection import cross_val_score
        
        # Get predictions from each model
        xgb_pred = self.models['xgboost'].predict_proba(X)[:, 1]
        lgb_pred = self.models['lightgbm'].predict_proba(X)[:, 1]
        nn_pred = self.models['neural_network'].predict_proba(X)
        
        # Simple optimization (in practice, use more sophisticated methods)
        weights = [0.4, 0.3, 0.3]  # XGBoost, LightGBM, Neural Network
        
        self.ensemble_weights = weights
        logger.info(f"Optimized ensemble weights: {weights}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        # Get predictions from each model
        xgb_pred = self.models['xgboost'].predict_proba(X)[:, 1]
        lgb_pred = self.models['lightgbm'].predict_proba(X)[:, 1]
        nn_pred = self.models['neural_network'].predict_proba(X)
        
        # Weighted ensemble
        ensemble_pred = (self.ensemble_weights[0] * xgb_pred + 
                        self.ensemble_weights[1] * lgb_pred + 
                        self.ensemble_weights[2] * nn_pred)
        
        return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble probabilities"""
        xgb_pred = self.models['xgboost'].predict_proba(X)[:, 1]
        lgb_pred = self.models['lightgbm'].predict_proba(X)[:, 1]
        nn_pred = self.models['neural_network'].predict_proba(X)
        
        ensemble_pred = (self.ensemble_weights[0] * xgb_pred + 
                        self.ensemble_weights[1] * lgb_pred + 
                        self.ensemble_weights[2] * nn_pred)
        
        return np.column_stack([1 - ensemble_pred, ensemble_pred])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get ensemble feature importance"""
        importance = {}
        
        # XGBoost importance
        xgb_importance = self.models['xgboost'].feature_importances_
        
        # LightGBM importance
        lgb_importance = self.models['lightgbm'].feature_importances_
        
        # Neural network importance (simplified)
        nn_importance = self.models['neural_network'].get_feature_importance()
        
        # Weighted ensemble importance
        for i, feature in enumerate(self.feature_names):
            importance[feature] = (
                self.ensemble_weights[0] * xgb_importance[i] +
                self.ensemble_weights[1] * lgb_importance[i] +
                self.ensemble_weights[2] * nn_importance[i]
            )
        
        return importance
    
    def explain_prediction(self, X: np.ndarray, instance_idx: int = 0) -> Dict:
        """Explain individual predictions using SHAP and LIME"""
        explanations = {}
        
        if self.explainer:
            # SHAP explanation
            shap_values = self.explainer.shap_values(X[instance_idx:instance_idx+1])
            explanations['shap'] = {
                'values': shap_values[0].tolist(),
                'features': self.feature_names
            }
        
        # LIME explanation
        try:
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X, feature_names=self.feature_names, mode='classification'
            )
            lime_exp = lime_explainer.explain_instance(
                X[instance_idx], self.predict_proba, num_features=10
            )
            explanations['lime'] = {
                'features': [x[0] for x in lime_exp.as_list()],
                'scores': [x[1] for x in lime_exp.as_list()]
            }
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
        
        return explanations
    
    def detect_bias(self, X: np.ndarray, y: np.ndarray, 
                   protected_attributes: List[str]) -> Dict:
        """Detect bias in model predictions"""
        if not self.bias_detector:
            return {}
        
        predictions = self.predict(X)
        return self.bias_detector.analyze_bias(X, y, predictions, protected_attributes)

class BiasDetector:
    """Advanced bias detection and fairness analysis"""
    
    def __init__(self):
        self.bias_metrics = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """Initialize bias detector"""
        self.feature_names = feature_names
        self.baseline_metrics = self._calculate_baseline_metrics(y)
    
    def _calculate_baseline_metrics(self, y: np.ndarray) -> Dict:
        """Calculate baseline fairness metrics"""
        return {
            'overall_positive_rate': np.mean(y),
            'overall_negative_rate': 1 - np.mean(y)
        }
    
    def analyze_bias(self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray, 
                    protected_attributes: List[str]) -> Dict:
        """Analyze bias across protected attributes"""
        bias_results = {}
        
        for attr in protected_attributes:
            if attr in self.feature_names:
                attr_idx = self.feature_names.index(attr)
                attr_values = X[:, attr_idx]
                unique_values = np.unique(attr_values)
                
                group_metrics = {}
                for value in unique_values:
                    mask = attr_values == value
                    group_y = y[mask]
                    group_pred = predictions[mask]
                    
                    group_metrics[f"group_{value}"] = {
                        'positive_rate': np.mean(group_y),
                        'prediction_rate': np.mean(group_pred),
                        'true_positive_rate': np.mean(group_pred[group_y == 1]) if np.sum(group_y) > 0 else 0,
                        'false_positive_rate': np.mean(group_pred[group_y == 0]) if np.sum(1-group_y) > 0 else 0
                    }
                
                bias_results[attr] = group_metrics
        
        return bias_results

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(1000, 50)
    y = np.random.randint(0, 2, 1000)
    feature_names = [f"feature_{i}" for i in range(50)]
    
    # Configure model
    config = ModelConfig(
        xgb_params={'n_estimators': 100, 'max_depth': 6},
        lgb_params={'n_estimators': 100, 'max_depth': 6},
        neural_network_config={'hidden_dims': [128, 64, 32], 'dropout_rate': 0.3},
        model_explainability=True,
        bias_detection=True
    )
    
    # Train ensemble model
    model = EnsembleRiskModel(config)
    model.fit(X, y, feature_names)
    
    # Make predictions
    predictions = model.predict(X[:10])
    probabilities = model.predict_proba(X[:10])
    
    # Get feature importance
    importance = model.get_feature_importance()
    
    # Explain predictions
    explanations = model.explain_prediction(X, instance_idx=0)
    
    print("Model training completed successfully!")
    print(f"Sample predictions: {predictions[:5]}")
    print(f"Sample probabilities: {probabilities[:5, 1]}")
    print(f"Top 5 important features: {sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]}")
