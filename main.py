import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import warnings
import torch as th
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import seaborn as sns # Added for correlation heatmap

# Ignore warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
STOCK_TICKER = "AAPL"
TRAIN_START_DATE = "2015-01-01"
TRAIN_END_DATE = "2021-12-31"
TEST_START_DATE = "2022-01-01"
TEST_END_DATE = "2023-12-31"

INITIAL_ACCOUNT_BALANCE = 10000
LOOKBACK_WINDOW_SIZE = 20
TRANSACTION_FEE_PERCENT = 0.0001 # Standard trading fee

# Position Size Limit
MAX_POSITION_PERCENTAGE = 0.9 # Allow up to 90% exposure

# Correlation Threshold
CORRELATION_THRESHOLD = 0.95

# Hyperparameter Suggestions (Manual Tuning Encouraged)
DQN_HYPERPARAMS = {
    "learning_rate": 5e-5,       # Suggestion: Try 1e-4, 5e-5, 1e-5
    "buffer_size": 100000,       # Suggestion: Try 50k, 100k, 200k
    "learning_starts": 2000,     # More steps before learning, ensure buffer is diverse
    "batch_size": 64,
    "gamma": 0.99,
    "target_update_interval": 600, # Suggestion: Try 500, 600, 1000
    "train_freq": 4,
    "gradient_steps": 1,
    "exploration_fraction": 0.2, # Suggestion: Try 0.1, 0.2, 0.3 (explore longer)
    "exploration_final_eps": 0.05 # Suggestion: Try 0.02, 0.05, 0.1
}

# Reward Function Weights (Manual Tuning Encouraged)
REWARD_SCALING = {
    'sharpe_weight': 0.15,       # Suggestion: Try 0.1, 0.15, 0.25
    'volatility_penalty': 0.05,  # Suggestion: Try 0, 0.05, 0.1
    'trade_penalty': 1.0,        # Suggestion: Try 0.5, 1.0, 2.0
    'volatility_threshold': 0.015 # Suggestion: Try 0.01, 0.015, 0.02
}

# --- ADX Calculation Helper (Handles MultiIndex) ---
def calculate_adx_multi(df_orig, high_col, low_col, close_col, period=14):
    """Calculate Average Directional Index (ADX) using specified column names/tuples"""
    df = df_orig.copy() # Work on a copy
    alpha = 1 / period

    # True Range
    df['TR'] = np.maximum(
        df[high_col] - df[low_col], # Use specific columns
        np.maximum(
            abs(df[high_col] - df[close_col].shift(1)),
            abs(df[low_col] - df[close_col].shift(1))
        )
    )

    # Directional Movement
    df['DMplus'] = np.where(
        (df[high_col] - df[high_col].shift(1)) > (df[low_col].shift(1) - df[low_col]),
        np.maximum(df[high_col] - df[high_col].shift(1), 0),
        0
    )
    df['DMminus'] = np.where(
        (df[low_col].shift(1) - df[low_col]) > (df[high_col] - df[high_col].shift(1)),
        np.maximum(df[low_col].shift(1) - df[low_col], 0),
        0
    )

    # Calculate Smoothed TR, DM+, DM- using Exponential Moving Average (EMA)
    # Use different name for intermediate ATR to avoid potential conflicts
    df['ATR_adx'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['DMplus_smooth'] = df['DMplus'].ewm(alpha=alpha, adjust=False).mean()
    df['DMminus_smooth'] = df['DMminus'].ewm(alpha=alpha, adjust=False).mean()

    # Directional Indicators
    df['DIplus'] = 100 * (df['DMplus_smooth'] / (df['ATR_adx'] + 1e-9))
    df['DIminus'] = 100 * (df['DMminus_smooth'] / (df['ATR_adx'] + 1e-9))

    # Directional Index
    df['DX'] = 100 * abs(df['DIplus'] - df['DIminus']) / (abs(df['DIplus'] + df['DIminus']) + 1e-9)

    # Average Directional Index
    ADX = df['DX'].ewm(alpha=alpha, adjust=False).mean()

    # Don't drop intermediate cols from the original df passed in, just return ADX Series
    return ADX


# --- Feature Engineering (Handles MultiIndex) ---
def add_indicators(df):
    """Add technical indicators to the dataframe, handling MultiIndex columns"""
    # --- Get the Ticker symbol and define column accessors ---
    if isinstance(df.columns, pd.MultiIndex):
        # Assumes only one ticker in the MultiIndex for simplicity
        ticker = df.columns.get_level_values('Ticker')[0]
        print(f"Detected MultiIndex columns, using ticker: {ticker}")
        adj_close_col = ('Adj Close', ticker)
        high_col = ('High', ticker)
        low_col = ('Low', ticker)
        close_col = ('Close', ticker)
        volume_col = ('Volume', ticker)
        open_col = ('Open', ticker) # Keep track even if not used directly below
    else:
        # Assume flat columns if not MultiIndex
        print("Assuming flat columns (no MultiIndex).")
        adj_close_col = 'Adj Close'
        high_col = 'High'
        low_col = 'Low'
        close_col = 'Close'
        volume_col = 'Volume'
        open_col = 'Open'

        # Check if required columns exist
        required_cols = [adj_close_col, high_col, low_col, close_col, volume_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
             raise KeyError(f"Missing required columns for flat structure: {missing_cols}")


    # --- Basic Price Features ---
    # Use the correctly selected Series; new columns get simple string names
    df['Returns'] = df[adj_close_col].pct_change()
    df['Log_Returns'] = np.log(df[adj_close_col] / df[adj_close_col].shift(1))

    # --- Moving Averages ---
    df['SMA_5'] = df[adj_close_col].rolling(window=5).mean()
    df['SMA_10'] = df[adj_close_col].rolling(window=10).mean()
    df['SMA_20'] = df[adj_close_col].rolling(window=20).mean()
    df['SMA_50'] = df[adj_close_col].rolling(window=50).mean()
    df['EMA_12'] = df[adj_close_col].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df[adj_close_col].ewm(span=26, adjust=False).mean()

    # --- Moving Average Ratios / Relative Position ---
    # Now dividing Series by Series
    df['Price_to_SMA_20'] = df[adj_close_col] / df['SMA_20']
    df['Price_to_SMA_50'] = df[adj_close_col] / df['SMA_50']
    df['SMA5_SMA20_Ratio'] = df['SMA_5'] / df['SMA_20']

    # --- Momentum Indicators ---
    df['ROC_5'] = df[adj_close_col].pct_change(periods=5)
    df['ROC_10'] = df[adj_close_col].pct_change(periods=10)
    df['ROC_20'] = df[adj_close_col].pct_change(periods=20)

    # MACD (uses EMA columns which are Series)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # RSI (Relative Strength Index)
    delta = df[adj_close_col].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    avg_gain = up.ewm(com=13, adjust=False).mean()
    avg_loss = abs(down.ewm(com=13, adjust=False).mean())
    rs = avg_gain / (avg_loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))

    # --- Volatility Indicators ---
    # Bollinger Bands
    df['BB_Middle'] = df['SMA_20'] # Use existing Series
    df['BB_Std'] = df[adj_close_col].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-9)
    df['BB_Position'] = (df[adj_close_col] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)

    # Average True Range (ATR)
    # Calculate intermediate columns locally, don't add to df unless needed
    atr_hl = df[high_col] - df[low_col]
    atr_hc = abs(df[high_col] - df[close_col].shift(1))
    atr_lc = abs(df[low_col] - df[close_col].shift(1))
    tr = pd.concat([atr_hl, atr_hc, atr_lc], axis=1).max(axis=1, skipna=False)
    df['ATR_14'] = tr.ewm(com=13, adjust=False).mean()

    # --- Volume Indicators ---
    df['Volume_Change'] = df[volume_col].pct_change()
    df['Volume_MA_20'] = df[volume_col].rolling(window=20).mean()
    df['Volume_Ratio'] = df[volume_col] / (df['Volume_MA_20'] + 1e-9)

    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df[adj_close_col].diff()).fillna(0) * df[volume_col]).cumsum()
    df['OBV_Change'] = df['OBV'].pct_change()

    # ADX - Average Directional Index
    df['ADX'] = calculate_adx_multi(df, high_col, low_col, close_col, period=14)

    # --- Identify feature groups ---
    # Use the simple string names of the generated features
    all_gen_features = [
        'Returns', 'Log_Returns', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'Price_to_SMA_20', 'Price_to_SMA_50', 'SMA5_SMA20_Ratio',
        'ROC_5', 'ROC_10', 'ROC_20',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'RSI',
        'BB_Middle', 'BB_Std', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
        'ATR_14',
        'Volume_Change', 'Volume_MA_20', 'Volume_Ratio',
        'OBV', 'OBV_Change',
        'ADX',
    ]
    # Filter out features that might have failed calculation (e.g., if input data was missing)
    all_gen_features = [f for f in all_gen_features if f in df.columns]

    technical_features = [
        f for f in all_gen_features if 'Volume' not in f and 'OBV' not in f
    ]
    volume_related_features = [
        f for f in all_gen_features if 'Volume' in f or 'OBV' in f
    ]
    # Add the original Volume column name (might be tuple or string)
    # The normalizer needs to handle this; let's assume it uses simple names for now
    # Or, better, ensure the normalizer uses the names from feature_groups

    feature_groups = {
        'technical': technical_features,
        'volume': volume_related_features
    }

    # --- Final Cleanup ---
    # Drop intermediate calculation columns only if they were added (TR wasn't directly added)
    # df.drop(['HL', 'HC', 'LC', 'TR'], axis=1, errors='ignore', inplace=True) # TR not added directly
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    # Drop rows with remaining NaNs (important before normalization/env)
    # This might shorten the dataframe, especially at the beginning
    initial_len = len(df)
    df.dropna(inplace=True)
    if len(df) < initial_len:
        print(f"Dropped {initial_len - len(df)} rows containing NaNs after indicator calculation.")

    return df, feature_groups


# --- TensorBoard Callback (Placeholder - Standard SB3 Logging often sufficient) ---
class TensorboardCallback(BaseCallback):
    """Custom callback for logging additional metrics to TensorBoard (Optional)"""
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
    def _on_training_start(self): pass
    def _on_step(self) -> bool:
        # Standard SB3 logs reward, episode length, etc.
        # Custom logging (e.g., net worth) might require VecEnv wrapper modifications.
        return True


# --- Custom Neural Network Architecture ---
class CNNLSTMFeatureExtractor(BaseFeaturesExtractor):
    """ Hybrid CNN-LSTM feature extractor """
    def __init__(self, observation_space, features_dim=128):
        super(CNNLSTMFeatureExtractor, self).__init__(observation_space, features_dim)

        n_features = observation_space.shape[0]
        portfolio_features = 3  # shares, cash, net_worth (normalized)

        if n_features <= portfolio_features:
             raise ValueError(f"Observation space dimension ({n_features}) is too small for portfolio features ({portfolio_features}).")

        market_features_total = n_features - portfolio_features
        if market_features_total <= 0:
             raise ValueError("Observation space suggests no market features are present.")

        # Ensure the market features part is divisible by lookback
        if market_features_total % LOOKBACK_WINDOW_SIZE != 0:
            raise ValueError(f"Market features dimension ({market_features_total}) not divisible by lookback window ({LOOKBACK_WINDOW_SIZE}). Check feature calculation or lookback size.")

        features_per_step = market_features_total // LOOKBACK_WINDOW_SIZE
        self.lookback = LOOKBACK_WINDOW_SIZE
        self.features_per_step = features_per_step
        self.portfolio_features = portfolio_features
        print(f"CNNLSTMFeatureExtractor: Input features per step: {features_per_step}, Lookback: {self.lookback}")


        # CNN layers: Input shape (N, C_in, L_in) = (batch*lookback, 1, features_per_step)
        # OR (N, features_per_step, lookback)? Let's assume processing features over time.
        # Input shape: [batch, features_per_step, lookback]
        self.cnn = nn.Sequential(
            # Conv1d expects (batch, channels, length)
            # Here, channels = features_per_step, length = lookback
            nn.Conv1d(in_channels=self.features_per_step, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=1, padding=1), # Careful with pooling over time dimension
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1) # Output: (batch, 128, 1) -> Squeeze -> (batch, 128)
        )
        cnn_output_size = 128

        # LSTM layer: Input shape (N, L, H_in) = (batch, lookback, lstm_input_size)
        # How to get sequence? Treat CNN output as features for LSTM?
        # Let's reconsider: Maybe CNN should process each step independently.
        # Alternative CNN structure: process each step
        # self.cnn_step = nn.Sequential(...) # Takes (batch*lookback, features_per_step) -> (batch*lookback, cnn_step_output_size)
        # Then reshape to (batch, lookback, cnn_step_output_size) for LSTM

        # Sticking with original idea: CNN summarizes the lookback window's features
        self.lstm = nn.LSTM(
            input_size=cnn_output_size, # Treat CNN output as input feature vector
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )
        lstm_output_size = 128

        # Final layers combining LSTM output and portfolio features
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_size + portfolio_features, 128),
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        batch_size = observations.shape[0]

        # Extract market and portfolio features
        market_features = observations[:, :-self.portfolio_features]
        portfolio = observations[:, -self.portfolio_features:]

        # Reshape market features for CNN: [batch, lookback * features_per_step] -> [batch, features_per_step, lookback]
        market_features_cnn = market_features.view(batch_size, self.lookback, self.features_per_step).permute(0, 2, 1)

        # Apply CNN: [batch, features_per_step, lookback] -> [batch, cnn_output_size, 1] -> [batch, cnn_output_size]
        cnn_out = self.cnn(market_features_cnn).squeeze(-1)

        # Prepare for LSTM: Treat CNN output as a single feature vector for the sequence.
        # Create an artificial sequence dimension of length 1.
        lstm_input = cnn_out.unsqueeze(1) # Shape: [batch, 1, cnn_output_size]

        # Process with LSTM: [batch, 1, cnn_output_size] -> [batch, 1, lstm_output_size]
        lstm_out, _ = self.lstm(lstm_input)

        # Take the output of the only step: [batch, 1, lstm_output_size] -> [batch, lstm_output_size]
        lstm_out_last = lstm_out.squeeze(1)

        # Concatenate LSTM output with portfolio state
        combined = th.cat((lstm_out_last, portfolio), dim=1) # Shape: [batch, lstm_output_size + portfolio_features]

        # Final fully connected layers
        out = self.fc(combined) # Shape: [batch, features_dim]

        return out


# --- Feature Normalizer ---
class FeatureNormalizer:
    """Handles normalization of different feature groups"""
    def __init__(self, feature_groups):
        self.feature_groups = feature_groups
        self.scalers = {
            'technical': MinMaxScaler(feature_range=(-1, 1)), # Min-max for technical indicators
            'volume': StandardScaler()  # Z-score for volume metrics
        }
        self.is_fitted = False
        self.fitted_columns = {} # Store columns fitted for each group (simple string names)

    def fit(self, df):
        """Fit scalers to the training data using simple column names"""
        print("Fitting normalizer...")
        for group, features in self.feature_groups.items():
            # Assumes 'features' contains simple string names generated by add_indicators
            valid_features = [f for f in features if f in df.columns]
            if valid_features:
                print(f"Fitting group '{group}' with {len(valid_features)} features.")
                # Ensure data being fit is numeric and finite
                data_to_fit = df[valid_features].replace([np.inf, -np.inf], np.nan).dropna()
                if data_to_fit.empty:
                     print(f"Warning: No valid numeric data to fit scaler for group '{group}' after cleaning infinities/NaNs.")
                     continue
                try:
                    self.scalers[group].fit(data_to_fit)
                    self.fitted_columns[group] = valid_features # Store fitted columns
                except Exception as e:
                     print(f"Error fitting scaler for group '{group}': {e}")
                     print(f"Data sample:\n{data_to_fit.head()}")

            else:
                 print(f"Warning: No valid features found for group '{group}' in the dataframe during fitting.")
        self.is_fitted = True
        print("Normalizer fitting complete.")


    def transform(self, df):
        """Transform features using fitted scalers"""
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")

        transformed_df = df.copy()
        print("Transforming data with normalizer...")

        for group, scaler in self.scalers.items():
            if group in self.fitted_columns:
                transform_cols = [f for f in self.fitted_columns[group] if f in transformed_df.columns]
                if transform_cols:
                    # Handle potential infinities before transforming
                    transformed_df[transform_cols] = transformed_df[transform_cols].replace([np.inf, -np.inf], 0)
                    try:
                        transformed_data = scaler.transform(transformed_df[transform_cols])
                        transformed_df[transform_cols] = transformed_data
                    except Exception as e:
                         print(f"Error transforming group '{group}': {e}")
                         # Handle error, e.g., fill with zeros or skip
                         transformed_df[transform_cols] = 0 # Example fallback
                else:
                     print(f"Warning: No fitted columns for group '{group}' found in the dataframe during transform.")
            else:
                print(f"Warning: Scaler for group '{group}' was not fitted or had no columns.")

        return transformed_df

    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)


# --- Enhanced Stock Trading Environment (Handles MultiIndex Price Access) ---
class EnhancedStockTradingEnv(gym.Env):
    """Enhanced stock trading environment with improvements and MultiIndex handling"""
    metadata = {'render_modes': ['human']}

    def __init__(self, df, feature_groups, normalizer, # Normalizer should be passed fitted
                 original_prices,
                 initial_balance=INITIAL_ACCOUNT_BALANCE,
                 lookback_window_size=LOOKBACK_WINDOW_SIZE,
                 transaction_fee_percent=TRANSACTION_FEE_PERCENT,
                 reward_params=REWARD_SCALING,
                 allow_fractional_shares=True,
                 max_position_percentage=MAX_POSITION_PERCENTAGE):
        super(EnhancedStockTradingEnv, self).__init__()

        self.df = df.copy() # Should receive NORMALIZED data
        self.feature_groups = feature_groups
        self.normalizer = normalizer # Store the passed normalizer
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.transaction_fee_percent = transaction_fee_percent
        self.reward_params = reward_params
        self.allow_fractional_shares = allow_fractional_shares
        self.max_position_value_ratio = max_position_percentage

        self.orig_prices = original_prices.copy()
        self.orig_prices = self.orig_prices.reindex(self.df.index).ffill().bfill()

        # --- Store the correct Adj Close column name from the original data ---
        # This requires the original column structure info if df is already normalized
        # Let's infer it based on typical yfinance output structure
        # We need the original df structure to get the price, let's add it as arg or find a workaround
        # Workaround: Assume the normalized df still has index compatible with an original price series
        # We will need to pass the original price series separately for lookups
        # --- For simplicity now, assume df passed *still contains* the original price column ---
        # --- This means normalization should happen *inside* the env or be handled carefully ---
        # --- Revised Approach: Pass original prices separately ---
        # Let's revert: Assume df passed to env is NOT normalized yet. Normalize inside.
        # --- No, stick to passing normalized data as planned. Need price source. ---
        # --> Add original price series as an argument

        # Consolidate all features expected by the normalizer (simple names)
        self.all_features = []
        for group in self.feature_groups:
             if group in self.normalizer.fitted_columns:
                  self.all_features.extend(self.normalizer.fitted_columns[group])
        self.all_features = sorted(list(set(self.all_features)))

        if not self.all_features:
             raise ValueError("No features found after normalizer setup. Check feature generation and normalization.")
        print(f"Environment using {len(self.all_features)} market features.")

        # Actions: 0: Hold, 1-4: Buy %, 5-8: Sell %
        self.action_space = spaces.Discrete(9)

        # Observation space: Lookback * n_features + 3 portfolio features
        n_market_features = len(self.all_features)
        obs_shape = (self.lookback_window_size * n_market_features + 3,)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        print(f"Observation space shape: {obs_shape}")

        # Internal state
        self.current_step_index = 0 # Use index position instead of step count
        self.shares_held = 0.0
        self.cash_in_hand = 0.0
        self.net_worth = 0.0
        self.trade_history = []
        self.portfolio_values = []
        self.daily_returns = []
        self.last_action = 0


    def _get_current_price(self):
        """Gets the original adjusted close price for the current step's date"""
        current_date = self.df.index[self.current_step_index]
        # Use .get() with default to handle potential missing dates if needed
        price = self.orig_prices.get(current_date)
        if price is None or not np.isfinite(price):
            # Fallback: Use the last known price from the main df (might be normalized)
            # Or better: use last known orig_price
            price = self.orig_prices.iloc[self.current_step_index] # Less safe if index jumps
            print(f"Warning: Could not find original price for {current_date}, using iloc lookup.")
            if price is None or not np.isfinite(price):
                 # Final fallback - very unlikely if data processed correctly
                 return self.initial_balance # Arbitrary safe value? No, better to use last good value
                 # Find last valid price
                 last_valid_price = self.orig_prices.iloc[:self.current_step_index+1].dropna().iloc[-1]
                 print(f"Using last valid price: {last_valid_price}")
                 return last_valid_price

        return price

    def _next_observation(self):
        # Get market data for the lookback window from NORMALIZED df
        frame_start = max(0, self.current_step_index - self.lookback_window_size + 1)
        frame_end = self.current_step_index + 1 # Inclusive of current step

        frame_data = self.df.iloc[frame_start:frame_end]

        # Pad if not enough history at the start
        actual_len = len(frame_data)
        if actual_len < self.lookback_window_size:
            padding_size = self.lookback_window_size - actual_len
            # Pad with the first row's data
            padding_df = pd.concat([frame_data.iloc[[0]]] * padding_size, ignore_index=False)
            frame_data = pd.concat([padding_df, frame_data], ignore_index=False)


        # Select features (already normalized)
        market_obs_df = frame_data[self.all_features]
        obs_market = market_obs_df.values.flatten()

        # Portfolio state (normalized)
        current_price = self._get_current_price() # Get original price
        initial_price = self.orig_prices.iloc[0] # Price at very start
        # Normalize shares relative to initial potential holding capacity
        norm_shares = self.shares_held / (self.initial_balance / (initial_price + 1e-9) + 1e-9)
        norm_cash = self.cash_in_hand / self.initial_balance
        norm_net_worth = self.net_worth / self.initial_balance

        portfolio_info = np.array([norm_shares, norm_cash, norm_net_worth])

        # Combine
        obs = np.concatenate([obs_market, portfolio_info]).astype(np.float32)

        # Check shape consistency
        if obs.shape != self.observation_space.shape:
            # print(f"Debug Obs Shape: Market={obs_market.shape}, Portfolio={portfolio_info.shape}")
            raise ValueError(f"Observation shape mismatch: expected {self.observation_space.shape}, got {obs.shape}. Market features={len(self.all_features)}, Lookback={self.lookback_window_size}")

        return obs

    def calculate_reward(self, previous_value, current_value):
        # (Reward calculation logic unchanged)
        profit_loss = current_value - previous_value
        reward = profit_loss # Start with simple P/L

        history_len = 30
        if len(self.portfolio_values) >= history_len:
            recent_values = np.array(self.portfolio_values[-history_len:])
            returns = np.diff(recent_values) / (recent_values[:-1] + 1e-9)

            avg_return = np.mean(returns)
            std_return = np.std(returns)

            sharpe_component = 0
            if std_return > 1e-9:
                 sharpe_component = avg_return / std_return
            # Scale Sharpe effect by magnitude of P/L? Or add directly? Let's add directly for stability.
            # reward += self.reward_params['sharpe_weight'] * sharpe_component
            # --> Simpler: Modulate P/L reward by Sharpe
            reward = profit_loss * (1 + self.reward_params['sharpe_weight'] * np.tanh(sharpe_component)) # Use tanh to bound sharpe effect

            if std_return > self.reward_params['volatility_threshold']:
                # Penalty scales with P/L magnitude - only penalize negative P/L more? Or overall volatility?
                # Let's use a simple penalty factor
                 reward *= (1 - self.reward_params['volatility_penalty'])


        # Trade Penalty (applied regardless of P/L)
        if self.last_action != 0: # If Buy or Sell action was attempted/executed
            reward -= self.reward_params['trade_penalty']

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cash_in_hand = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0.0
        # Start simulation after the lookback window number of steps IN THE INDEX
        self.current_step_index = self.lookback_window_size # Use index position
        self.trade_history = []
        self.portfolio_values = [self.initial_balance] * 30 # Initialize history
        self.daily_returns = [0.0] * 30
        self.last_action = 0

        # Ensure first observation is valid
        if self.current_step_index >= len(self.df.index):
             raise ValueError("Lookback window size exceeds dataframe length after dropping NaNs.")

        observation = self._next_observation()
        info = {'net_worth': self.net_worth,
                'shares_held': self.shares_held,
                'cash_in_hand': self.cash_in_hand}

        return observation, info

    def step(self, action):
        self.last_action = action
        current_price = self._get_current_price() # Get original price for trading
        previous_net_worth = self.net_worth

        action_str = ''
        shares_traded = 0.0
        trade_cost = 0.0

        # Apply Position Size Limit Logic
        max_allowed_position_value = self.net_worth * self.max_position_value_ratio
        current_position_value = self.shares_held * current_price

        if action >= 1 and action <= 4: # Buy actions
            buy_percentage = action * 0.25
            max_affordable_shares = self.cash_in_hand / (current_price * (1 + self.transaction_fee_percent) + 1e-9)
            shares_to_buy_attempt = max_affordable_shares * buy_percentage

            max_additional_value = max(0, max_allowed_position_value - current_position_value)
            max_shares_by_limit = max_additional_value / (current_price * (1 + self.transaction_fee_percent) + 1e-9)

            shares_to_buy = min(shares_to_buy_attempt, max_shares_by_limit)

            if not self.allow_fractional_shares:
                shares_to_buy = np.floor(shares_to_buy)

            if shares_to_buy > 1e-6:
                action_str_detail = f'{shares_to_buy:.4f}' if self.allow_fractional_shares else f'{int(shares_to_buy)}'
                action_str = f'BUY {action_str_detail}'
                if shares_to_buy < shares_to_buy_attempt * 0.99:
                     action_str += ' (Limit Hit)'

                shares_traded = shares_to_buy
                cost = shares_to_buy * current_price
                fee = cost * self.transaction_fee_percent  # Calculate fee
                total_spent = cost + fee  # Total cost including fee
                self.cash_in_hand -= total_spent
                self.shares_held += shares_to_buy
                

                self.trade_history.append((self.df.index[self.current_step_index], action_str, shares_traded, self.net_worth, fee))
            else:
                action_str = 'HOLD (cannot afford/limit BUY)'

        elif action >= 5 and action <= 8: # Sell actions
            sell_percentage = (action - 4) * 0.25
            shares_to_sell = self.shares_held * sell_percentage

            if not self.allow_fractional_shares:
                shares_to_sell = np.floor(shares_to_sell)

            if shares_to_sell > 1e-6:
                action_str_detail = f'{shares_to_sell:.4f}' if self.allow_fractional_shares else f'{int(shares_to_sell)}'
                action_str = f'SELL {action_str_detail}'
                shares_traded = -shares_to_sell
                revenue = shares_to_sell * current_price
                fee = revenue * self.transaction_fee_percent  # Calculate fee on revenue
                self.cash_in_hand += revenue - fee
                self.shares_held -= shares_to_sell

                # Record only the fee in trade history
                self.trade_history.append((self.df.index[self.current_step_index], action_str, shares_traded, self.net_worth, fee))
            else:
                action_str = 'HOLD (nothing to SELL)'

        else: # Hold
            action_str = 'HOLD'

        # Update net worth & history
        self.net_worth = self.cash_in_hand + self.shares_held * current_price
        self.portfolio_values.append(self.net_worth)
        daily_return = (self.net_worth / previous_net_worth) - 1 if previous_net_worth > 0 else 0
        self.daily_returns.append(daily_return)

        # Calculate reward
        reward = self.calculate_reward(previous_net_worth, self.net_worth)


        # Advance time step index
        self.current_step_index += 1

        # Check if done
        terminated = self.current_step_index >= len(self.df.index) - 1
        truncated = False # Not using time limits

        # Get next observation or zeros if done
        observation = self._next_observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        # Info dict
        info = {'net_worth': self.net_worth,
                'shares_held': self.shares_held,
                'cash_in_hand': self.cash_in_hand,
                'action_taken': action_str,
                'daily_return': daily_return}

        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})

            start_idx_render = self.lookback_window_size
            end_idx_render = min(self.current_step_index, len(self.df.index) - 1)

            if start_idx_render >= end_idx_render:
                 print("Not enough data points in simulation to render.")
                 plt.close(fig)
                 return

            # --- FIX: Filter plot dates to ensure they exist in orig_prices ---
            plot_dates_available = self.df.index[start_idx_render:end_idx_render+1]
            # Find dates present in both indices
            valid_plot_dates_mask = plot_dates_available.isin(self.orig_prices.index)
            plot_dates = plot_dates_available[valid_plot_dates_mask]

            if plot_dates.empty:
                 print("Warning: No overlapping dates found between simulation index and original prices for plotting.")
                 plt.close(fig)
                 return
            # --- End Fix ---

            # Use the filtered plot_dates for lookups
            price_data = self.orig_prices.loc[plot_dates] # Safe lookup now

            # Plot price and MAs
            ax1.plot(plot_dates, price_data.values, label='Stock Price', color='blue', alpha=0.7)
            # Make sure MA lookups also use filtered dates
            if 'SMA_20' in self.df.columns:
                ax1.plot(plot_dates, self.df['SMA_20'].loc[plot_dates], label='SMA 20', color='orange', alpha=0.5, linestyle='--')
            if 'SMA_50' in self.df.columns:
                ax1.plot(plot_dates, self.df['SMA_50'].loc[plot_dates], label='SMA 50', color='purple', alpha=0.5, linestyle='--')

            # Plot trades (filter trade dates as well)
            buy_trade_info = [(t[0], t[2]) for t in self.trade_history if 'BUY' in t[1]] # date, size
            sell_trade_info = [(t[0], t[2]) for t in self.trade_history if 'SELL' in t[1]]

            # Filter trades to only those dates within the plot_dates range and available in orig_prices
            valid_buy_trades = [(d, self.orig_prices.get(d, np.nan), s) for d, s in buy_trade_info if d in plot_dates]
            valid_sell_trades = [(d, self.orig_prices.get(d, np.nan), s) for d, s in sell_trade_info if d in plot_dates]

            # Further filter out any trades where price lookup failed (unlikely now but safe)
            valid_buys = [(d, p, s) for d, p, s in valid_buy_trades if pd.notna(p)]
            valid_sells = [(d, p, s) for d, p, s in valid_sell_trades if pd.notna(p)]


            if valid_buys:
                 buy_dates_plot, buy_prices_plot, buy_sizes_plot = zip(*valid_buys)
                 ax1.scatter(buy_dates_plot, buy_prices_plot, s=[abs(s)*20 + 20 for s in buy_sizes_plot], marker='^', color='green', label='Buy', alpha=0.8, edgecolors='k')

            if valid_sells:
                 sell_dates_plot, sell_prices_plot, sell_sizes_plot = zip(*valid_sells)
                 ax1.scatter(sell_dates_plot, sell_prices_plot, s=[abs(s)*20 + 20 for s in sell_sizes_plot], marker='v', color='red', label='Sell', alpha=0.8, edgecolors='k')


            ax1.set_title(f'{STOCK_TICKER} Trading Activity & Price')
            ax1.set_ylabel('Price ($)')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)

            # Plot portfolio value (filter portfolio dates)
            portfolio_dates_raw = [self.df.index[self.lookback_window_size]] + [t[0] for t in self.trade_history]
            portfolio_values_raw = [self.initial_balance] + [t[3] for t in self.trade_history]

            # Ensure dates are within the plot_dates (which are already filtered)
            valid_portfolio_indices = [i for i, d in enumerate(portfolio_dates_raw) if d in plot_dates]
            portfolio_dates_plot = [portfolio_dates_raw[i] for i in valid_portfolio_indices]
            portfolio_values_plot = [portfolio_values_raw[i] for i in valid_portfolio_indices]


            if portfolio_dates_plot:
                ax2.plot(portfolio_dates_plot, portfolio_values_plot, color='purple', label='Portfolio Value')
                ax2.axhline(y=self.initial_balance, color='gray', linestyle='--', label=f'Initial ${self.initial_balance:,.0f}')
                ax2.set_title('Portfolio Value Over Time')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Value ($)')
                ax2.legend(loc='upper left')
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            self._print_performance_metrics() # Print metrics after showing plot
    
    
    def _print_performance_metrics(self):
        # (Performance metric calculation unchanged, uses self.net_worth, self.trade_history)
        if not self.trade_history:
            print("No trades executed to calculate metrics.")
            return

        final_value = self.net_worth
        total_return_pct = ((final_value / self.initial_balance) - 1) * 100
        n_trades = sum(1 for t in self.trade_history if abs(t[2]) > 1e-6)
        total_fees = sum(t[4] for t in self.trade_history)

      # Buy & Hold using original prices over the simulation period
        sim_start_date_potential = self.df.index[self.lookback_window_size]
        sim_end_date_potential = self.df.index[min(self.current_step_index, len(self.df.index) - 1)]
        
        try:
            start_indexer = self.orig_prices.index.get_indexer([sim_start_date_potential], method='nearest')
            end_indexer = self.orig_prices.index.get_indexer([sim_end_date_potential], method='nearest')

            # Ensure indexer didn't return -1 (not found, unlikely with 'nearest' unless index empty)
            if start_indexer[0] == -1 or end_indexer[0] == -1:
                raise IndexError("Nearest date lookup failed, index might be empty.")

            # Get the actual dates using the integer positions
            sim_start_date = self.orig_prices.index[start_indexer[0]]
            sim_end_date = self.orig_prices.index[end_indexer[0]]

            start_price = self.orig_prices.loc[sim_start_date] # Use .loc for label-based lookup
            end_price = self.orig_prices.loc[sim_end_date]
        except Exception as e:
            print(f"Warning: Error getting Buy & Hold start/end prices: {e}")
            sim_start_date = sim_start_date_potential # Fallback for printing
            sim_end_date = sim_end_date_potential   # Fallback for printing
            start_price = np.nan
            end_price = np.nan
        # --- End Fix ---
        if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
            buy_hold_return_pct = ((end_price / start_price) - 1) * 100
        else:
            buy_hold_return_pct = 0
            print("Warning: Could not calculate Buy & Hold return accurately due to missing prices.")


        # Risk Metrics
        portfolio_hist = np.array(self.portfolio_values)
        if len(portfolio_hist) < 2:
             sharpe, sortino, max_drawdown = 0, 0, 0
        else:
             returns_hist = np.diff(portfolio_hist) / (portfolio_hist[:-1] + 1e-9)
             annual_factor = 252
             mean_return_ann = np.mean(returns_hist) * annual_factor
             std_return_ann = np.std(returns_hist) * np.sqrt(annual_factor)

             sharpe = mean_return_ann / (std_return_ann + 1e-9)

             downside_returns = returns_hist[returns_hist < 0]
             sortino = 0
             if len(downside_returns) > 0:
                 downside_std_ann = np.std(downside_returns) * np.sqrt(annual_factor)
                 if downside_std_ann > 1e-9:
                     sortino = mean_return_ann / downside_std_ann

             peak = np.maximum.accumulate(portfolio_hist)
             drawdown = (peak - portfolio_hist) / (peak + 1e-9)
             max_drawdown = np.max(drawdown) * 100


        print(f"\n--- Episode Performance Metrics ---")
        print(f"Simulation Period: {sim_start_date.date()} to {sim_end_date.date()}")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return_pct:.2f}%")
        print(f"Buy & Hold Return: {buy_hold_return_pct:.2f}%")
        print(f"Strategy vs Buy & Hold: {total_return_pct - buy_hold_return_pct:.2f}%")
        print(f"Number of Trades: {n_trades}")
        print(f"Total Fees Paid: ${total_fees:,.2f}")
        print(f"Sharpe Ratio (Annualized): {sharpe:.2f}")
        print(f"Sortino Ratio (Annualized): {sortino:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"------------------------------------")


# --- Simple Baseline Strategy: MA Crossover (Handles MultiIndex) ---
def simulate_ma_crossover(df_orig, # Pass original df with prices and indicators
                          short_window, long_window, initial_balance, fee_percent):
    """Simulates MA Crossover, handling potential MultiIndex in price columns"""
    print(f"\n--- Simulating MA Crossover ({short_window}/{long_window}) Baseline ---")
    df = df_orig.copy() # Work on a copy

    # --- Detect column type ---
    if isinstance(df.columns, pd.MultiIndex):
        ticker = df.columns.get_level_values('Ticker')[0]
        adj_close_col = ('Adj Close', ticker)
    else:
        adj_close_col = 'Adj Close'

    # MA columns should have simple names from add_indicators
    short_ma_col = f'SMA_{short_window}'
    long_ma_col = f'SMA_{long_window}'

    # Check required columns
    required = [adj_close_col, short_ma_col, long_ma_col]
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Error: MA Crossover baseline missing required columns: {missing}")
        return 0, 0

    cash = initial_balance
    shares = 0.0
    position = 0 # 0 = out, 1 = in
    portfolio_value = initial_balance

    # Iterate using index for safe price lookup
    for i in range(1, len(df)):
        current_price = df[adj_close_col].iloc[i]
        if not pd.notna(current_price) or current_price <= 0: continue # Skip if price invalid

        sma_short = df[short_ma_col].iloc[i]
        sma_long = df[long_ma_col].iloc[i]
        prev_sma_short = df[short_ma_col].iloc[i-1]
        prev_sma_long = df[long_ma_col].iloc[i-1]

        # Check if MAs are valid
        if not (pd.notna(sma_short) and pd.notna(sma_long) and
                pd.notna(prev_sma_short) and pd.notna(prev_sma_long)):
            continue # Skip step if MA data is missing

        # Buy signal
        if prev_sma_short <= prev_sma_long and sma_short > sma_long and position == 0:
            shares_to_buy = cash / (current_price * (1 + fee_percent) + 1e-9)
            if shares_to_buy > 0:
                cost = shares_to_buy * current_price * (1 + fee_percent)
                cash -= cost
                shares += shares_to_buy
                position = 1

        # Sell signal
        elif prev_sma_short >= prev_sma_long and sma_short < sma_long and position == 1:
            if shares > 0:
                revenue = shares * current_price * (1 - fee_percent)
                cash += revenue
                shares = 0.0
                position = 0

        portfolio_value = cash + shares * current_price

    final_value = portfolio_value
    total_return_pct = ((final_value / initial_balance) - 1) * 100 if initial_balance > 0 else 0

    print(f"MA Crossover Final Value: ${final_value:,.2f}")
    print(f"MA Crossover Total Return: {total_return_pct:.2f}%")
    print(f"------------------------------------")
    return final_value, total_return_pct


# --- Multi-episode evaluation function ---
def evaluate_model_multiple_episodes(model, env_lambda, n_episodes=5):
    """Evaluate a model over multiple episodes using an environment creation lambda"""
    all_returns = []
    all_sharpes = []
    all_drawdowns = []
    all_trades = []

    print(f"\n--- Evaluating Model over {n_episodes} Episodes ---")

    base_seed = 42

    for i in range(n_episodes):
        print(f"Starting evaluation epsisone {i+1}/{n_episodes} with seed {base_seed + i}")

        # Create a fresh environment for each evaluation episode
        eval_env = env_lambda() # Use lambda to create instance
        obs, info = eval_env.reset(seed=base_seed + i)
        terminated = truncated = False
        ep_portfolio_values = [eval_env.initial_balance]
        ep_trades = 0

        while not terminated and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)

            ep_portfolio_values.append(info['net_worth'])
            # Count actual trades based on action string or shares traded
            if info.get('action_taken') not in ['HOLD', 'HOLD (cannot afford/limit BUY)', 'HOLD (nothing to SELL)']:
                 ep_trades += 1

        # Performance metrics (can call env's internal method if available)
        final_value = info['net_worth']
        ep_total_return_pct = ((final_value / eval_env.initial_balance) - 1) * 100
        all_returns.append(ep_total_return_pct)
        all_trades.append(ep_trades)

        # Calculate Sharpe and Drawdown
        portfolio_hist = np.array(ep_portfolio_values)
        if len(portfolio_hist) < 2:
             sharpe, max_drawdown = 0, 0
        else:
             returns_hist = np.diff(portfolio_hist) / (portfolio_hist[:-1] + 1e-9)
             annual_factor = 252
             mean_return_ann = np.mean(returns_hist) * annual_factor
             std_return_ann = np.std(returns_hist) * np.sqrt(annual_factor)
             sharpe = mean_return_ann / (std_return_ann + 1e-9)
             peak = np.maximum.accumulate(portfolio_hist)
             drawdown = (peak - portfolio_hist) / (peak + 1e-9)
             max_drawdown = np.max(drawdown) * 100

        all_sharpes.append(sharpe)
        all_drawdowns.append(max_drawdown)

        print(f"Episode {i+1}/{n_episodes} finished. Return: {ep_total_return_pct:.2f}%, Trades: {ep_trades}, Sharpe: {sharpe:.2f}, Max DD: {max_drawdown:.2f}%")
        # Clean up env? Usually not necessary unless heavy resources used
        # eval_env.close() # If env has a close method

    # Summary Stats
    mean_return = np.mean(all_returns)
    std_return = np.std(all_returns)
    mean_sharpe = np.mean(all_sharpes)
    mean_drawdown = np.mean(all_drawdowns)
    mean_trades = np.mean(all_trades)

    print("\n--- Multi-Episode Evaluation Summary ---")
    print(f"Average Return: {mean_return:.2f}% (Std: {std_return:.2f}%)")
    print(f"Average Sharpe Ratio: {mean_sharpe:.2f}")
    print(f"Average Max Drawdown: {mean_drawdown:.2f}%")
    print(f"Average Trades per Episode: {mean_trades:.1f}")
    print(f"--------------------------------------")

    return {
        'mean_return': mean_return, 'std_return': std_return,
        'mean_sharpe': mean_sharpe, 'mean_drawdown': mean_drawdown,
        'mean_trades': mean_trades
    }


# --- Main Execution ---
if __name__ == "__main__":
    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./tensorboard_logs", exist_ok=True)
    os.makedirs("./plots", exist_ok=True)

    # 1. Load Data
    print(f"Loading data for {STOCK_TICKER}...")
    data = yf.download(
        STOCK_TICKER,
        start=TRAIN_START_DATE,
        end=TEST_END_DATE,
        auto_adjust=False, # Keep 'Adj Close'
        progress=True
    )
    if data.empty:
        raise ValueError("No data downloaded. Check ticker or dates.")

    # Handle potential MultiIndex immediately after download
    if isinstance(data.columns, pd.MultiIndex):
        print("Flattening MultiIndex columns...")
        # Keep only the first level (Price type) - assumes single ticker download
        # Or better: create specific column names
        data.columns = ['_'.join(col).strip('_') for col in data.columns.values]
        # Rename columns more cleanly if needed, e.g., 'Adj Close_AAPL' -> 'Adj Close'
        # This depends on exact MultiIndex structure
        data = data.rename(columns={f'Adj Close_{STOCK_TICKER}': 'Adj Close',
                                      f'Close_{STOCK_TICKER}': 'Close',
                                      f'High_{STOCK_TICKER}': 'High',
                                      f'Low_{STOCK_TICKER}': 'Low',
                                      f'Open_{STOCK_TICKER}': 'Open',
                                      f'Volume_{STOCK_TICKER}': 'Volume'})
        print(f"Columns after flattening: {data.columns}")

    # Ensure required columns exist after potential flattening
    required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if not all(col in data.columns for col in required_ohlcv):
         raise ValueError(f"Missing required OHLCV columns after potential flattening. Found: {data.columns}")

    #Storing origanl prices before modifications
    if 'Adj Close' not in data.columns:
        raise ValueError("Missing 'Adj Close' column after loading/flattering.")
    original_adj_close = data['Adj Close'].copy()

    # 2. Add Indicators
    print("Adding technical indicators...")
    # Pass copy, add_indicators now expects flat column names
    data_with_indicators, feature_groups = add_indicators(data.copy())
    if data_with_indicators.empty:
         raise ValueError("DataFrame became empty after adding indicators. Check calculations/NaN handling.")
    
    # --- Split Data FIRST (for baseline) ---
    train_data_full_features = data_with_indicators.loc[TRAIN_START_DATE:TRAIN_END_DATE].copy()
    test_data_full_features = data_with_indicators.loc[TEST_START_DATE:TEST_END_DATE].copy()

    # Filter date ranges strictly for the baseline splits
    train_data_full_features = train_data_full_features[
        (train_data_full_features.index >= TRAIN_START_DATE) & (train_data_full_features.index <= TRAIN_END_DATE)
    ]
    test_data_full_features = test_data_full_features[
        (test_data_full_features.index >= TEST_START_DATE) & (test_data_full_features.index <= TEST_END_DATE)
    ]

    original_adj_close_train = original_adj_close.loc[train_data_full_features.index]
    original_adj_close_test = original_adj_close.loc[test_data_full_features.index]

    # --- Run Baseline HERE (using test_data_full_features) ---
    print("\n--- Simulating MA Crossover (20/50) Baseline ---")
    simulate_ma_crossover(
        df_orig=test_data_full_features, # Use data BEFORE feature removal
        short_window=20, long_window=50,
        initial_balance=INITIAL_ACCOUNT_BALANCE,
        fee_percent=TRANSACTION_FEE_PERCENT
    )

    # --- Correlation Analysis & Feature Removal (using simple column names) ---
    print("\nPerforming Correlation Analysis...")
    features_to_analyze = feature_groups.get('technical', []) + feature_groups.get('volume', [])
    features_to_analyze = [f for f in features_to_analyze if f in data_with_indicators.columns] # Ensure features exist

    if not features_to_analyze:
         print("Warning: No features available for correlation analysis.")
         features_to_drop = []
    else:
        correlation_matrix = data_with_indicators[features_to_analyze].corr().abs()
        upper_triangle_mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        upper_triangle = correlation_matrix.where(upper_triangle_mask)
        features_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > CORRELATION_THRESHOLD)]

        baseline_required_features = ['SMA_20', 'SMA_50']
        baseline_required_features = [f for f in baseline_required_features if f in data_with_indicators.columns]
        features_to_drop = [f for f in features_to_drop if f not in baseline_required_features]
        print(f"Preserved {len(baseline_required_features)} features required for baseline comparison")
            
        if features_to_drop:
            print(f"Removing {len(features_to_drop)} highly correlated features (Threshold > {CORRELATION_THRESHOLD}): {features_to_drop}")
            # Remove from the main dataframe being used going forward
            data_with_indicators.drop(columns=features_to_drop, inplace=True)

            # Update feature_groups dictionary
            for group in list(feature_groups.keys()): # Iterate over keys copy
                feature_groups[group] = [f for f in feature_groups[group] if f not in features_to_drop]
                if not feature_groups[group]: del feature_groups[group]
        else:
            print("No features found exceeding the correlation threshold.")

    # 3. Split Data (Chronological)
    # Use the dataframe that includes indicators and has correlated features removed
    train_data = data_with_indicators.loc[TRAIN_START_DATE:TRAIN_END_DATE].copy()
    test_data = data_with_indicators.loc[TEST_START_DATE:TEST_END_DATE].copy()

    # Filter date ranges strictly AFTER splitting to avoid lookahead bias during dropna etc.
    train_data = train_data[(train_data.index >= TRAIN_START_DATE) & (train_data.index <= TRAIN_END_DATE)]
    test_data = test_data[(test_data.index >= TEST_START_DATE) & (test_data.index <= TEST_END_DATE)]

    print(f"\nTraining data points after processing: {len(train_data)}")
    print(f"Testing data points after processing: {len(test_data)}")
    if len(train_data) < LOOKBACK_WINDOW_SIZE * 2 or len(test_data) < LOOKBACK_WINDOW_SIZE * 2:
         raise ValueError("Insufficient data in train or test split after processing.")


    # 4. Normalize Features (Fit on Train, Transform Train & Test)
    print("Normalizing features...")
    normalizer = FeatureNormalizer(feature_groups)
    # Important: Fit normalizer ONLY on training data features
    normalized_train_data = normalizer.fit_transform(train_data)
    # Transform test data using the SAME fitted normalizer
    normalized_test_data = normalizer.transform(test_data)

    # Add back original price columns needed by env? No, env fetches them.

    # 5. Create Environments using NORMALIZED data
    train_env_lambda = lambda: EnhancedStockTradingEnv(
        normalized_train_data, feature_groups, normalizer, # Pass fitted normalizer
        original_prices= original_adj_close_train,
        reward_params=REWARD_SCALING, allow_fractional_shares=True,
        max_position_percentage=MAX_POSITION_PERCENTAGE
    )
    test_env_lambda = lambda: EnhancedStockTradingEnv(
        normalized_test_data, feature_groups, normalizer, # Pass same normalizer
        original_prices= original_adj_close_test,
        reward_params=REWARD_SCALING, allow_fractional_shares=True,
        max_position_percentage=MAX_POSITION_PERCENTAGE
    )
    # Use DummyVecEnv
    # Check environment creation works before training
    try:
        check_env = DummyVecEnv([train_env_lambda])
        check_env.reset()
        print("Environment check passed.")
        check_env.close()
    except Exception as e:
        print(f"Error during environment check: {e}")
        raise e

    train_env = DummyVecEnv([train_env_lambda])
    test_env = DummyVecEnv([test_env_lambda])


    # 6. Define Model Policy Kwargs
    policy_kwargs = dict(
        features_extractor_class=CNNLSTMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=[128, 64]
    )

    # 7. Create DQN Agent
    tensorboard_log_path = "./tensorboard_logs/stock_trading_dqn_v3"
    print(f"\nInitializing DQN model...")
    print(f"Hyperparameters: {DQN_HYPERPARAMS}")
    print(f"Reward Weights: {REWARD_SCALING}")

    model = DQN(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=tensorboard_log_path,
        seed=42, # Set seed for reproducibility here if possible
        **DQN_HYPERPARAMS
    )

    # 8. Train the Agent
    TRAINING_SEED = 42
    TOTAL_TIMESTEPS = 100000 # Adjust as needed
    print(f"\nStarting training for {TOTAL_TIMESTEPS} timesteps with seed {TRAINING_SEED}...")
    print("Monitor training: tensorboard --logdir ./tensorboard_logs")
    callback = TensorboardCallback()
    # Seed the environment(s)
    train_env.seed(TRAINING_SEED)
    test_env.seed(TRAINING_SEED + 1) # Use different seed for test env if needed

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        log_interval=50,
        callback=callback,
        reset_num_timesteps=False,
        tb_log_name="DQN_run_v3"
    )
    print("Training finished.")

    # 9. Save Model
    MODEL_PATH = "./models/enhanced_dqn_stock_trader_v3"
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


    # --- Evaluation ---
    print("\n--- Evaluating Trained Agent on Test Data ---")

    # 10. Multi-Episode Evaluation for Robustness
    # Pass the lambda to create fresh environments for each run
    multi_ep_results = evaluate_model_multiple_episodes(model, test_env_lambda, n_episodes=5)

    # 11. Detailed Single Episode Evaluation & Rendering
    print("\n--- Detailed Evaluation & Rendering (Final Episode) ---")
    # Create one instance for detailed run and render
    final_eval_env = test_env_lambda()
    obs, info = final_eval_env.reset()
    terminated = truncated = False
    while not terminated and not truncated:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = final_eval_env.step(action)
    # Render the final state
    print("Rendering trading activity...")
    final_eval_env.render(mode='human')
    # Metrics are printed inside render call now

    print("\nEvaluation complete.")

    # Clean up environments
    train_env.close()
    test_env.close() # Although VecEnv might handle underlying env closure
    # final_eval_env.close() # If env had explicit close method