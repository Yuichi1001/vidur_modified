from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, RegressorMixin

from vidur.config import (
    BaseReplicaSchedulerConfig,
    LSTMExecutionTimePredictorConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.execution_time_predictor.sklearn_execution_time_predictor import (
    SklearnExecutionTimePredictor,
)


class TorchLSTMNet(nn.Module):
    def __init__(self, in_size: int, hidden: int, layers: int, dr: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_size,
            hidden_size=hidden,
            num_layers=layers,
            dropout=(dr if layers > 1 else 0.0),
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


class _TorchLSTMRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 20,
        device: Optional[str] = None,
        sequence_length: int = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.sequence_length = sequence_length
        self.random_state = random_state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # placeholders set during fit
        self._model: Optional[nn.Module] = None

    def _build_model(self, input_size: int) -> nn.Module:
        return TorchLSTMNet(input_size, self.hidden_size, self.num_layers, self.dropout)

    def fit(self, X: np.ndarray, y: np.ndarray):
        if isinstance(X, np.ndarray):
            features = X
        else:
            features = X.to_numpy()

        # shape to [N, T=1, F]
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        N, F = features.shape
        T = max(1, self.sequence_length)
        # Use degenerate sequence of length 1 by default
        seq = features.reshape(N, 1, F) if T == 1 else features.reshape(N, T, F // T)

        xt = torch.from_numpy(seq).float()
        # y may be a pandas Series; coerce to numpy
        y_np = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
        yt = torch.from_numpy(y_np.astype(np.float32))

        ds = TensorDataset(xt, yt)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        self._model = self._build_model(input_size=seq.shape[-1]).to(self.device)
        opt = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self._model.train()
        for _ in range(self.max_epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._model is not None, "Model not trained"
        self._model.eval()

        if isinstance(X, np.ndarray):
            features = X
        else:
            features = X.to_numpy()

        if features.ndim == 1:
            features = features.reshape(-1, 1)
        N, F = features.shape
        T = max(1, self.sequence_length)
        seq = features.reshape(N, 1, F) if T == 1 else features.reshape(N, T, F // T)

        with torch.no_grad():
            preds: np.ndarray = (
                self._model(torch.from_numpy(seq).float().to(self.device))
                .cpu()
                .numpy()
            )
        return preds

    # Ensure sklearn/pickle can serialize the estimator by saving only state_dict
    def __getstate__(self):
        state = self.__dict__.copy()
        if self._model is not None:
            # move weights to cpu for portable serialization
            cpu_state = {k: v.detach().cpu() for k, v in self._model.state_dict().items()}
            state['_model_state_dict'] = cpu_state
            state['_model_input_size'] = self._model.lstm.input_size
            state['_model'] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        sd = state.get('_model_state_dict')
        if sd is not None:
            in_size = state.get('_model_input_size', self.input_size)
            self._model = self._build_model(in_size)
            self._model.load_state_dict(sd)
            self._model = self._model.to(self.device)


class LSTMExecutionTimePredictor(SklearnExecutionTimePredictor):
    def __init__(
        self,
        predictor_config: LSTMExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )

    def _get_grid_search_params(self):
        # Map config lists into sklearn grid for our torch regressor
        return {
            "hidden_size": self._config.hidden_size,
            "num_layers": self._config.num_layers,
            "dropout": self._config.dropout,
            "learning_rate": self._config.learning_rate,
            "batch_size": self._config.batch_size,
            "max_epochs": self._config.max_epochs,
        }

    def _get_estimator(self):
        # input_size is resolved at fit-time from X.shape; we pass a placeholder
        return _TorchLSTMRegressor(device="cpu")


