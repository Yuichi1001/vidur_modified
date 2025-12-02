from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from vidur.config import (
    BaseReplicaSchedulerConfig,
    MetaLearningExecutionTimePredictorConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.execution_time_predictor.linear_regression_execution_time_predictor import (
    LinearRegressionExecutionTimePredictor,
)
from vidur.logger import init_logger

logger = init_logger(__name__)


class MetaPredictorWrapper(BaseEstimator):
    """
    A wrapper that combines the source model and the meta-learner (mapping model).
    """

    def __init__(self, source_model: BaseEstimator, meta_model: BaseEstimator):
        self.source_model = source_model
        self.meta_model = meta_model

    def predict(self, X):
        # 1. Predict using the source model (trained on A100)
        source_pred = self.source_model.predict(X)

        # 2. Use the source prediction as the feature for the meta model
        # The meta model (LinearRegression) expects a 2D array (n_samples, n_features)
        X_meta = source_pred.reshape(-1, 1)

        # 3. Predict the final target time
        return self.meta_model.predict(X_meta)


class MetaLearningExecutionTimePredictor(LinearRegressionExecutionTimePredictor):
    def __init__(
            self,
            predictor_config: MetaLearningExecutionTimePredictorConfig,
            replica_config: ReplicaConfig,
            replica_scheduler_config: BaseReplicaSchedulerConfig,
            metrics_config: MetricsConfig,
    ) -> None:
        self._metric_logs = []  # Store metrics for aggregation
        super().__init__(
            predictor_config=predictor_config,
            replica_config=replica_config,
            replica_scheduler_config=replica_scheduler_config,
            metrics_config=metrics_config,
        )
        # After all training is done (in super init), log the summary
        self._log_overall_metrics()

    def _log_overall_metrics(self):
        if not self._metric_logs:
            return

        logger.info("=" * 40)
        logger.info("Meta Learning Overall Performance Summary")
        logger.info("=" * 40)

        avg_mape = np.mean([m["mape"] for m in self._metric_logs])
        avg_rmse = np.mean([m["rmse"] for m in self._metric_logs])
        avg_r2 = np.mean([m["r2"] for m in self._metric_logs])

        logger.info(f"Average MAPE: {avg_mape:.4f}%")
        logger.info(f"Average RMSE: {avg_rmse:.4f}")
        logger.info(f"Average R2:   {avg_r2:.4f}")
        logger.info("=" * 40)

    def _get_source_file_path(self, target_file_path: str) -> str:
        """
        Derive the source device file path from the target device file path.
        Assumes the path contains the device name.
        """
        target_device = self._replica_config.device
        source_device = self._config.source_device

        # Simple string replacement.
        # We assume the directory structure is .../compute/{device}/...
        # If target_device is 'a40' and source is 'a100', we replace 'a40' with 'a100'.
        if target_device not in target_file_path:
            logger.warning(
                f"Target device {target_device} not found in path {target_file_path}. "
                f"Cannot derive source path."
            )
            return target_file_path

        return target_file_path.replace(target_device, source_device)

    def _load_compute_df_for_device(self, file_path: str) -> pd.DataFrame:
        """Helper to load compute DF with standard filtering (copied from parent logic)"""
        df = self._read_input_file(file_path)

        # Apply the same filtering logic as in SklearnExecutionTimePredictor._load_compute_df
        # Note: We assume the source device data has the same model config columns
        # (n_head, n_embd, etc.) as the target, which is true for the same model.

        df = df[
            (df["n_head"] == self._model_config.num_q_heads)
            & (df["n_kv_head"] == self._model_config.num_kv_heads)
            & (df["n_embd"] == self._model_config.embedding_dim)
            & (df["n_expanded_embd"] == self._model_config.mlp_hidden_dim)
            & (df["use_gated_mlp"] == self._model_config.use_gated_mlp)
            & (df["vocab_size"] == self._model_config.vocab_size)
            # Note: For tensor parallel size, we assume the source data was collected
            # with the same TP degree or we might need to adjust.
            # Usually A100 and A40 profiles for same model use same TP if possible,
            # but if not, this filter might be too strict or wrong.
            # For now, assume TP matches.
            & (
                    df["num_tensor_parallel_workers"]
                    == self._replica_config.tensor_parallel_size
            )
            ]

        for column in [
            "time_stats.post_attention_layernorm.median",
            "time_stats.add.median",
            "time_stats.input_layernorm.median",
        ]:
            if column not in df.columns:
                df[column] = 0
            else:
                df.fillna({column: 0}, inplace=True)

        return df

    def _get_estimator(self) -> BaseEstimator:
        if self._config.source_model_type == "random_forrest":
            return RandomForestRegressor()
        return make_pipeline(PolynomialFeatures(), LinearRegression())

    def _get_meta_estimator(self) -> BaseEstimator:
        if self._config.meta_model_type == "random_forrest":
            # Using the same config as source model for simplicity,
            # but could be separated if needed.
            return RandomForestRegressor(
                n_estimators=self._config.num_estimators[0] if isinstance(self._config.num_estimators,
                                                                          list) else self._config.num_estimators,
                max_depth=self._config.max_depth[0] if isinstance(self._config.max_depth,
                                                                  list) else self._config.max_depth,
                min_samples_split=self._config.min_samples_split[0] if isinstance(self._config.min_samples_split,
                                                                                  list) else self._config.min_samples_split
            )
        return LinearRegression()

    def _get_grid_search_params(self) -> Dict[str, Any]:
        if self._config.source_model_type == "random_forrest":
            return {
                "n_estimators": self._config.num_estimators,
                "max_depth": self._config.max_depth,
                "min_samples_split": self._config.min_samples_split,
            }
        return {
            "polynomialfeatures__degree": self._config.polynomial_degree,
            "polynomialfeatures__include_bias": self._config.polynomial_include_bias,
            "polynomialfeatures__interaction_only": self._config.polynomial_interaction_only,
            "linearregression__fit_intercept": self._config.fit_intercept,
        }

    def _train_meta_model_instance(
            self,
            model_name: str,
            target_df: pd.DataFrame,
            source_df: pd.DataFrame,
            feature_cols: List[str],
            target_col: str,
    ) -> BaseEstimator:

        # 1. Train Source Model (A100)
        source_model = self._get_estimator()
        grid_search_params = self._get_grid_search_params()

        if len(source_df) == 0:
            logger.warning(
                f"Source data for {model_name} is empty. Falling back to target-only training."
            )
            return self._train_model(model_name, target_df, feature_cols, target_col)

        if len(source_df) < self._config.k_fold_cv_splits:
            cv = 2
        else:
            cv = self._config.k_fold_cv_splits

        grid_search = GridSearchCV(
            estimator=source_model,
            param_grid=grid_search_params,
            scoring=self._get_scorer(),
            cv=cv,
            n_jobs=self._config.num_training_job_threads,
        )
        grid_search.fit(source_df[feature_cols], source_df[target_col])
        best_source_model = grid_search.best_estimator_

        # 2. Sample Target Data (A40)
        fraction = self._config.target_data_fraction
        if 0 < fraction < 1.0:
            # Use a fixed random state for reproducibility
            sampled_target_df = target_df.sample(frac=fraction, random_state=42)
        else:
            sampled_target_df = target_df

        if len(sampled_target_df) == 0:
            raise ValueError(f"Sampled target data for {model_name} is empty.")

        # 3. Get predictions from source model on the sampled target data points
        source_preds = best_source_model.predict(sampled_target_df[feature_cols])

        # 4. Train the Meta Learner (Mapping: Source Pred -> Target Actual)
        # We use a simple Linear Regression without polynomial features
        # because the non-linearity is captured by the source model.
        meta_model = self._get_meta_estimator()
        X_meta = source_preds.reshape(-1, 1)
        y_meta = sampled_target_df[target_col]

        meta_model.fit(X_meta, y_meta)

        if isinstance(meta_model, LinearRegression):
            logger.info(
                f"Trained Meta Model for {model_name}. "
                f"Source Data: {len(source_df)}, Target Data (Sampled): {len(sampled_target_df)}. "
                f"Mapping: y_target = {meta_model.coef_[0]:.4f} * y_source + {meta_model.intercept_:.4f}"
            )
        else:
            logger.info(
                f"Trained Meta Model ({self._config.meta_model_type}) for {model_name}. "
                f"Source Data: {len(source_df)}, Target Data (Sampled): {len(sampled_target_df)}."
            )

        # 5. Wrap them together
        final_model = MetaPredictorWrapper(best_source_model, meta_model)

        # Cache logic is tricky because we need to cache the WRAPPER,
        # but the parent class's _train_model handles caching.
        # We are bypassing _train_model here, so we should handle caching or
        # delegate to a method that does.
        # Since we are overriding _train_compute_models, we are responsible for calling the training
        # and storing it.

        # Let's manually cache it here using the helper from parent
        # We need a hash that represents both source and target data ideally,
        # but sticking to target df hash is probably "good enough" for cache invalidation
        # if we assume config changes (like source device) change the hash too.
        model_hash = self._get_model_hash(model_name, sampled_target_df)
        self._store_model_in_cache(model_name, model_hash, final_model)

        # Also store predictions for debugging/analysis if needed
        # Construct a DF that has the features and the prediction
        # MODIFIED: Use the FULL target_df for validation/reporting, not just the training sample
        debug_df = target_df.copy()
        debug_df["prediction"] = final_model.predict(debug_df[feature_cols])

        # Calculate error on full dataset for logging
        try:
            mape = self.mean_absolute_percentage_error(debug_df[target_col], debug_df["prediction"])

            # RMSE
            mse = np.mean((debug_df[target_col] - debug_df["prediction"]) ** 2)
            rmse = np.sqrt(mse)

            # R2 Score (using sklearn implementation if available, or manual)
            # R2 = 1 - u/v, where u = sum((y_true - y_pred) ** 2), v = sum((y_true - y_true.mean()) ** 2)
            ss_res = np.sum((debug_df[target_col] - debug_df["prediction"]) ** 2)
            ss_tot = np.sum((debug_df[target_col] - debug_df[target_col].mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

            # Store metrics for overall summary
            self._metric_logs.append({
                "model": model_name,
                "mape": mape,
                "rmse": rmse,
                "r2": r2
            })

            logger.info(
                f"Meta Model {model_name} Full Validation Data Metrics:\n"
                f"  MAPE: {mape:.4f}%\n"
                f"  RMSE: {rmse:.4f}\n"
                f"  R2:   {r2:.4f}"
            )
        except Exception as e:
            logger.warning(f"Failed to calculate metrics for {model_name}: {e}")

        # We can use parent's _store_training_prediction_data but it expects the model to have .predict
        # Our wrapper has .predict, so it works.
        self._store_training_prediction_data(
            model_name,
            model_hash,
            debug_df,
            feature_cols,
            target_col,
            final_model,
        )

        return final_model

    def _train_compute_models(self) -> Dict[str, BaseEstimator]:
        # Load Target Data
        target_df = self._load_compute_df_for_device(self._compute_input_file)
        target_df = self._get_compute_df_with_derived_features(target_df)

        # Load Source Data
        source_file = self._get_source_file_path(self._compute_input_file)
        source_df = self._load_compute_df_for_device(source_file)
        source_df = self._get_compute_df_with_derived_features(source_df)

        models = {}
        model_names = [
            "attn_pre_proj",
            "attn_post_proj",
            "mlp_up_proj",
            "mlp_down_proj",
            "mlp_act",
            "input_layernorm",
            "post_attention_layernorm",
            "attn_rope",
            "add",
        ]

        for model_name in model_names:
            models[model_name] = self._train_meta_model_instance(
                model_name=model_name,
                target_df=target_df,
                source_df=source_df,
                feature_cols=["num_tokens"],
                target_col=f"time_stats.{model_name}.median",
            )

        # Handle Attention Models (separate file)
        # We'll do that in _train_attention_layer_models override

        # Handle Collective Models (All Reduce, Send Recv)
        # Typically network depends on bandwidth, meta learning across different networks
        # (e.g. IB vs RoCE or different speeds) is valid but maybe less critical than compute.
        # For simplicity, we will use STANDARD training for network ops
        # unless we want to meta-learn them too.
        # Let's stick to standard for network for now to reduce complexity,
        # or just use target data (since network might be very different).

        if self._replica_config.num_pipeline_stages > 1:
            # Fallback to standard training for network
            send_recv_df = self._load_send_recv_df(self._send_recv_input_file)
            send_recv_df = self._get_send_recv_df_with_derived_features(
                send_recv_df
            )
            models["send_recv"] = self._train_model(
                "send_recv",
                send_recv_df,
                ["num_tokens"],
                "time_stats.send_recv.median",
            )

        if self._replica_config.tensor_parallel_size > 1:
            all_reduce_df = self._load_all_reduce_df(self._all_reduce_input_file)
            all_reduce_df = self._get_all_reduce_df_with_derived_features(
                all_reduce_df
            )
            models["all_reduce"] = self._train_model(
                "all_reduce",
                all_reduce_df,
                ["num_tokens"],
                "time_stats.all_reduce.median",
            )

        return models

    def _load_attention_df_for_device(self, file_path: str) -> pd.DataFrame:
        """Helper for attention DF loading"""
        df = pd.read_csv(file_path)
        df = df.drop_duplicates()

        for column in [
            "time_stats.attn_kv_cache_save.median",
        ]:
            if column not in df.columns:
                df[column] = 0
            else:
                df.fillna({column: 0}, inplace=True)

        return df[
            (df["n_embd"] == self._model_config.embedding_dim)
            & (df["n_q_head"] == self._model_config.num_q_heads)
            & (df["n_kv_head"] == self._model_config.num_kv_heads)
            & (df["block_size"] == self._block_size)
            & (
                    df["num_tensor_parallel_workers"]
                    == self._replica_config.tensor_parallel_size
            )
            ]

    def _train_attention_layer_models(self) -> Dict[str, BaseEstimator]:
        # Target
        target_df = self._load_attention_df_for_device(self._attention_input_file)
        target_df = self._get_attention_df_with_derived_features(target_df)

        # Source
        source_file = self._get_source_file_path(self._attention_input_file)
        source_df = self._load_attention_df_for_device(source_file)
        source_df = self._get_attention_df_with_derived_features(source_df)

        # KV Cache Save Model
        models = {}
        models["attn_kv_cache_save"] = self._train_meta_model_instance(
            "attn_kv_cache_save",
            target_df,
            source_df,
            ["num_tokens"],
            f"time_stats.attn_kv_cache_save.median",
        )

        # Prefill / Decode Separation
        t_prefill = target_df[~target_df["is_decode"]]
        t_decode = target_df[target_df["is_decode"]]

        s_prefill = source_df[~source_df["is_decode"]]
        s_decode = source_df[source_df["is_decode"]]

        # Attention Prefill
        models["attn_prefill"] = self._train_meta_model_instance(
            "attn_prefill",
            t_prefill,
            s_prefill,
            ["kv_cache_size", "prefill_chunk_size_squared"],
            "time_stats.attn_prefill.median",
        )

        # Attention Decode
        models["attn_decode"] = self._train_meta_model_instance(
            "attn_decode",
            t_decode,
            s_decode,
            ["batch_size", "kv_cache_size"],
            "time_stats.attn_decode.median",
        )

        return models
