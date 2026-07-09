"""
Forecasting Models for ERPsim Copilot
Includes: MA, WMA, Exponential Smoothing, Linear Regression
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from datetime import datetime, timedelta

from utils.logger import setup_logger


logger = setup_logger(__name__)


class ForecastModel:
    """Base class for forecast models"""
    
    def __init__(self, data: pd.Series, horizon: int = 30):
        """
        Initialize forecast model
        
        Args:
            data: Time series data (values only)
            horizon: Number of periods to forecast
        """
        self.data = data.values
        self.horizon = horizon
        self.forecast = None
        self.mape = None
        self.rmse = None
    
    def calculate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def calculate_rmse(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Root Mean Squared Error"""
        return np.sqrt(np.mean((actual - predicted) ** 2))
    
    def forecast_values(self) -> np.ndarray:
        """Generate forecast values"""
        raise NotImplementedError


class MovingAverageModel(ForecastModel):
    """Simple Moving Average Forecast"""
    
    def __init__(self, data: pd.Series, horizon: int = 30, window: int = 7):
        """
        Initialize MA model
        
        Args:
            data: Time series data
            horizon: Forecast horizon
            window: Moving average window size
        """
        super().__init__(data, horizon)
        self.window = min(window, len(self.data))
    
    def forecast_values(self) -> np.ndarray:
        """Generate forecast using simple moving average"""
        try:
            # Calculate moving average
            ma = np.convolve(self.data, np.ones(self.window) / self.window, mode='valid')
            last_ma = ma[-1] if len(ma) > 0 else self.data[-1]
            
            # Forecast using last MA value
            self.forecast = np.full(self.horizon, last_ma)
            
            # Calculate accuracy metrics using last 30% of data as test set
            test_size = max(5, len(self.data) // 3)
            train_data = self.data[:-test_size]
            test_data = self.data[-test_size:]
            
            ma_train = np.convolve(train_data, np.ones(self.window) / self.window, mode='valid')
            predictions = np.full(test_size, ma_train[-1] if len(ma_train) > 0 else train_data[-1])
            
            self.mape = self.calculate_mape(test_data, predictions[:test_size])
            self.rmse = self.calculate_rmse(test_data, predictions[:test_size])
            
            logger.info(f"MA Forecast - MAPE: {self.mape:.2f}%, RMSE: {self.rmse:.2f}")
            
            return self.forecast
            
        except Exception as e:
            logger.error(f"Error in MA forecast: {str(e)}")
            return np.full(self.horizon, self.data[-1])


class WeightedMovingAverageModel(ForecastModel):
    """Weighted Moving Average Forecast"""
    
    def __init__(self, data: pd.Series, horizon: int = 30, window: int = 7):
        """
        Initialize WMA model
        
        Args:
            data: Time series data
            horizon: Forecast horizon
            window: WMA window size
        """
        super().__init__(data, horizon)
        self.window = min(window, len(self.data))
    
    def forecast_values(self) -> np.ndarray:
        """Generate forecast using weighted moving average"""
        try:
            # Calculate weights (linear increasing)
            weights = np.arange(1, self.window + 1)
            weights = weights / weights.sum()
            
            # Calculate WMA
            wma_values = []
            for i in range(len(self.data) - self.window + 1):
                wma = np.sum(self.data[i:i + self.window] * weights)
                wma_values.append(wma)
            
            last_wma = wma_values[-1] if wma_values else self.data[-1]
            
            # Forecast
            self.forecast = np.full(self.horizon, last_wma)
            
            # Calculate accuracy
            test_size = max(5, len(self.data) // 3)
            train_data = self.data[:-test_size]
            test_data = self.data[-test_size:]
            
            wma_train = []
            for i in range(len(train_data) - self.window + 1):
                wma = np.sum(train_data[i:i + self.window] * weights)
                wma_train.append(wma)
            
            predictions = np.full(test_size, wma_train[-1] if wma_train else train_data[-1])
            
            self.mape = self.calculate_mape(test_data, predictions[:test_size])
            self.rmse = self.calculate_rmse(test_data, predictions[:test_size])
            
            logger.info(f"WMA Forecast - MAPE: {self.mape:.2f}%, RMSE: {self.rmse:.2f}")
            
            return self.forecast
            
        except Exception as e:
            logger.error(f"Error in WMA forecast: {str(e)}")
            return np.full(self.horizon, self.data[-1])


class ExponentialSmoothingModel(ForecastModel):
    """Exponential Smoothing Forecast"""
    
    def __init__(self, data: pd.Series, horizon: int = 30, alpha: float = 0.3):
        """
        Initialize Exponential Smoothing model
        
        Args:
            data: Time series data
            horizon: Forecast horizon
            alpha: Smoothing parameter (0-1)
        """
        super().__init__(data, horizon)
        self.alpha = alpha
    
    def forecast_values(self) -> np.ndarray:
        """Generate forecast using exponential smoothing"""
        try:
            # Apply exponential smoothing
            s = np.zeros(len(self.data))
            s[0] = self.data[0]
            
            for t in range(1, len(self.data)):
                s[t] = self.alpha * self.data[t] + (1 - self.alpha) * s[t - 1]
            
            last_smoothed = s[-1]
            
            # Forecast using last smoothed value
            self.forecast = np.full(self.horizon, last_smoothed)
            
            # Calculate accuracy
            test_size = max(5, len(self.data) // 3)
            train_data = self.data[:-test_size]
            test_data = self.data[-test_size:]
            
            s_train = np.zeros(len(train_data))
            s_train[0] = train_data[0]
            
            for t in range(1, len(train_data)):
                s_train[t] = self.alpha * train_data[t] + (1 - self.alpha) * s_train[t - 1]
            
            predictions = np.full(test_size, s_train[-1])
            
            self.mape = self.calculate_mape(test_data, predictions[:test_size])
            self.rmse = self.calculate_rmse(test_data, predictions[:test_size])
            
            logger.info(f"Exponential Smoothing Forecast - MAPE: {self.mape:.2f}%, RMSE: {self.rmse:.2f}")
            
            return self.forecast
            
        except Exception as e:
            logger.error(f"Error in Exponential Smoothing forecast: {str(e)}")
            return np.full(self.horizon, self.data[-1])


class LinearRegressionModel(ForecastModel):
    """Linear Regression Forecast"""
    
    def __init__(self, data: pd.Series, horizon: int = 30):
        """
        Initialize Linear Regression model
        
        Args:
            data: Time series data
            horizon: Forecast horizon
        """
        super().__init__(data, horizon)
        self.model = None
    
    def forecast_values(self) -> np.ndarray:
        """Generate forecast using linear regression"""
        try:
            # Prepare training data
            X = np.arange(len(self.data)).reshape(-1, 1)
            y = self.data
            
            # Train model
            self.model = LinearRegression()
            self.model.fit(X, y)
            
            # Generate forecast
            forecast_X = np.arange(len(self.data), len(self.data) + self.horizon).reshape(-1, 1)
            self.forecast = self.model.predict(forecast_X)
            
            # Ensure no negative values for quantity-like data
            self.forecast = np.maximum(self.forecast, 0)
            
            # Calculate accuracy
            test_size = max(5, len(self.data) // 3)
            train_data = self.data[:-test_size]
            test_data = self.data[-test_size:]
            
            X_train = np.arange(len(train_data)).reshape(-1, 1)
            model_train = LinearRegression()
            model_train.fit(X_train, train_data)
            
            X_test = np.arange(len(train_data), len(self.data)).reshape(-1, 1)
            predictions = model_train.predict(X_test)
            predictions = np.maximum(predictions, 0)
            
            self.mape = self.calculate_mape(test_data, predictions[:test_size])
            self.rmse = self.calculate_rmse(test_data, predictions[:test_size])
            
            logger.info(f"Linear Regression Forecast - MAPE: {self.mape:.2f}%, RMSE: {self.rmse:.2f}")
            
            return self.forecast
            
        except Exception as e:
            logger.error(f"Error in Linear Regression forecast: {str(e)}")
            return np.full(self.horizon, self.data[-1])


class ForecastEngine:
    """Ensemble forecasting engine"""
    
    def __init__(self, horizon: int = 30):
        """
        Initialize forecast engine
        
        Args:
            horizon: Forecast horizon in days
        """
        self.horizon = horizon
        self.models: Dict[str, ForecastModel] = {}
        self.best_model = None
        self.best_mape = float('inf')
    
    def fit(self, data: pd.Series) -> Dict[str, Any]:
        """
        Fit all models and select best one
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary with forecast and model details
        """
        try:
            if len(data) < 5:
                logger.warning("Insufficient data for forecasting")
                return self._create_result(data, None, "insufficient_data")
            
            # Initialize and fit all models
            self.models = {
                "moving_average": MovingAverageModel(data, self.horizon, window=7),
                "weighted_moving_average": WeightedMovingAverageModel(data, self.horizon, window=7),
                "exponential_smoothing": ExponentialSmoothingModel(data, self.horizon, alpha=0.3),
                "linear_regression": LinearRegressionModel(data, self.horizon)
            }
            
            # Fit all models
            for name, model in self.models.items():
                try:
                    model.forecast_values()
                except Exception as e:
                    logger.warning(f"Error fitting {name}: {str(e)}")
            
            # Select best model based on MAPE
            self.best_model = min(
                (m for m in self.models.values() if m.mape is not None),
                key=lambda m: m.mape,
                default=None
            )
            
            if self.best_model is None:
                logger.warning("No valid models fitted")
                return self._create_result(data, None, "no_valid_models")
            
            self.best_mape = self.best_model.mape
            
            return self._create_result(data, self.best_model, "success")
            
        except Exception as e:
            logger.error(f"Error in forecast engine: {str(e)}")
            return self._create_result(data, None, "error")
    
    def _create_result(self, data: pd.Series, model: ForecastModel, status: str) -> Dict[str, Any]:
        """Create forecast result dictionary"""
        result = {
            "status": status,
            "horizon": self.horizon,
            "data_points": len(data),
            "forecast": None,
            "best_model": None,
            "accuracy": None,
            "model_details": {}
        }
        
        if model:
            result["forecast"] = model.forecast.tolist()
            result["best_model"] = next(
                (k for k, v in self.models.items() if v is model),
                None
            )
            result["accuracy"] = {
                "mape": round(model.mape, 2),
                "rmse": round(model.rmse, 2)
            }
        
        # Add all model details
        for name, m in self.models.items():
            if m.mape is not None:
                result["model_details"][name] = {
                    "mape": round(m.mape, 2),
                    "rmse": round(m.rmse, 2)
                }
        
        return result
