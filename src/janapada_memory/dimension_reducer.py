"""
Dimension Reduction Module

Handles PCA-based dimension reduction from 3072-dimensional legacy embeddings
to 768-dimensional embeddings for BigQuery VECTOR operations.
"""

import os
import logging
import numpy as np
import pickle
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)


class DimensionReducer:
    """Handles dimension reduction for vector embeddings using PCA."""
    
    def __init__(
        self,
        target_dimensions: int = 768,
        preserve_variance: float = 0.95,
        scale_features: bool = True,
        model_path: Optional[str] = None
    ):
        """
        Initialize dimension reducer.
        
        Args:
            target_dimensions: Target number of dimensions
            preserve_variance: Minimum variance to preserve (0-1)
            scale_features: Whether to scale features before PCA
            model_path: Path to save/load trained PCA model
        """
        self.target_dimensions = target_dimensions
        self.preserve_variance = preserve_variance
        self.scale_features = scale_features
        self.model_path = model_path or "models/pca_reducer.pkl"
        
        self.pca_model = None
        self.scaler = None
        self.is_fitted = False
        self.variance_explained = None
        self.training_stats = {}
        
        # Create models directory if needed
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        logger.info(f"DimensionReducer initialized: {self.target_dimensions} dimensions, {self.preserve_variance:.1%} variance")

    def fit(
        self,
        embeddings: Union[List[List[float]], np.ndarray],
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Fit PCA model on training embeddings.
        
        Args:
            embeddings: Training embeddings (N x original_dims)
            save_model: Whether to save the fitted model
            
        Returns:
            Training results and statistics
        """
        logger.info("Fitting PCA dimension reduction model...")
        start_time = datetime.now()
        
        # Convert to numpy array
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        if len(embeddings.shape) != 2:
            raise ValueError("Embeddings must be 2D array (N x dimensions)")
        
        original_dims = embeddings.shape[1]
        n_samples = embeddings.shape[0]
        
        logger.info(f"Training data: {n_samples} samples, {original_dims} dimensions")
        
        # Feature scaling if enabled
        if self.scale_features:
            self.scaler = StandardScaler()
            embeddings_scaled = self.scaler.fit_transform(embeddings)
            logger.info("Applied feature scaling")
        else:
            embeddings_scaled = embeddings
            self.scaler = None
        
        # Fit PCA model
        # Start with target dimensions, but allow variance preservation to override
        initial_components = min(self.target_dimensions, original_dims, n_samples)
        
        pca_temp = PCA(n_components=initial_components)
        pca_temp.fit(embeddings_scaled)
        
        # Check variance preservation
        cumsum_variance = np.cumsum(pca_temp.explained_variance_ratio_)
        components_for_variance = np.searchsorted(cumsum_variance, self.preserve_variance) + 1
        
        # Use the minimum of target dimensions and variance preservation requirement
        final_components = min(self.target_dimensions, components_for_variance)
        
        logger.info(f"Components needed for {self.preserve_variance:.1%} variance: {components_for_variance}")
        logger.info(f"Using {final_components} components")
        
        # Fit final PCA model
        self.pca_model = PCA(n_components=final_components)
        reduced_embeddings = self.pca_model.fit_transform(embeddings_scaled)
        
        self.is_fitted = True
        self.variance_explained = np.sum(self.pca_model.explained_variance_ratio_)
        
        # Calculate training statistics
        training_time = (datetime.now() - start_time).total_seconds()
        
        self.training_stats = {
            "original_dimensions": original_dims,
            "target_dimensions": self.target_dimensions,
            "actual_dimensions": final_components,
            "n_training_samples": n_samples,
            "variance_explained": float(self.variance_explained),
            "variance_per_component": self.pca_model.explained_variance_ratio_.tolist(),
            "training_time_seconds": training_time,
            "compression_ratio": original_dims / final_components,
            "feature_scaling": self.scale_features,
            "trained_at": datetime.now().isoformat()
        }
        
        logger.info(f"PCA training completed in {training_time:.2f}s")
        logger.info(f"Variance explained: {self.variance_explained:.1%}")
        logger.info(f"Compression ratio: {self.training_stats['compression_ratio']:.1f}x")
        
        # Save model if requested
        if save_model:
            self.save_model()
        
        return self.training_stats

    def transform(
        self,
        embeddings: Union[List[List[float]], np.ndarray]
    ) -> np.ndarray:
        """
        Transform embeddings to reduced dimensions.
        
        Args:
            embeddings: Input embeddings to transform
            
        Returns:
            Reduced dimension embeddings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")
        
        # Convert to numpy array
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        # Apply scaling if used during training
        if self.scaler is not None:
            embeddings_scaled = self.scaler.transform(embeddings)
        else:
            embeddings_scaled = embeddings
        
        # Apply PCA transformation
        reduced_embeddings = self.pca_model.transform(embeddings_scaled)
        
        return reduced_embeddings

    def fit_transform(
        self,
        embeddings: Union[List[List[float]], np.ndarray],
        save_model: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fit model and transform embeddings in one step.
        
        Args:
            embeddings: Input embeddings
            save_model: Whether to save fitted model
            
        Returns:
            Tuple of (reduced_embeddings, training_stats)
        """
        training_stats = self.fit(embeddings, save_model=save_model)
        reduced_embeddings = self.transform(embeddings)
        
        return reduced_embeddings, training_stats

    def inverse_transform(
        self,
        reduced_embeddings: Union[List[List[float]], np.ndarray]
    ) -> np.ndarray:
        """
        Inverse transform from reduced to original dimensions (approximate).
        
        Args:
            reduced_embeddings: Reduced dimension embeddings
            
        Returns:
            Reconstructed original dimension embeddings
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before inverse transformation")
        
        if isinstance(reduced_embeddings, list):
            reduced_embeddings = np.array(reduced_embeddings)
        
        if len(reduced_embeddings.shape) == 1:
            reduced_embeddings = reduced_embeddings.reshape(1, -1)
        
        # Inverse PCA transformation
        reconstructed = self.pca_model.inverse_transform(reduced_embeddings)
        
        # Inverse scaling if used
        if self.scaler is not None:
            reconstructed = self.scaler.inverse_transform(reconstructed)
        
        return reconstructed

    def evaluate_quality(
        self,
        original_embeddings: Union[List[List[float]], np.ndarray],
        reduced_embeddings: Optional[np.ndarray] = None,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Evaluate dimension reduction quality.
        
        Args:
            original_embeddings: Original high-dimensional embeddings
            reduced_embeddings: Reduced embeddings (computed if None)
            sample_size: Sample size for similarity evaluation
            
        Returns:
            Quality metrics
        """
        if isinstance(original_embeddings, list):
            original_embeddings = np.array(original_embeddings)
        
        if reduced_embeddings is None:
            reduced_embeddings = self.transform(original_embeddings)
        
        # Sample data for efficiency
        n_samples = len(original_embeddings)
        if n_samples > sample_size:
            indices = np.random.choice(n_samples, sample_size, replace=False)
            orig_sample = original_embeddings[indices]
            reduced_sample = reduced_embeddings[indices]
        else:
            orig_sample = original_embeddings
            reduced_sample = reduced_embeddings
        
        logger.info(f"Evaluating reduction quality on {len(orig_sample)} samples")
        
        # Reconstruction error
        reconstructed = self.inverse_transform(reduced_sample)
        reconstruction_error = np.mean(np.square(orig_sample - reconstructed))
        
        # Similarity preservation
        orig_similarities = cosine_similarity(orig_sample)
        reduced_similarities = cosine_similarity(reduced_sample)
        
        # Correlation between similarity matrices
        orig_flat = orig_similarities[np.triu_indices_from(orig_similarities, k=1)]
        reduced_flat = reduced_similarities[np.triu_indices_from(reduced_similarities, k=1)]
        similarity_correlation = np.corrcoef(orig_flat, reduced_flat)[0, 1]
        
        # Neighbor preservation (top-k)
        neighbor_preservation = self._evaluate_neighbor_preservation(
            orig_sample, reduced_sample, k=10
        )
        
        quality_metrics = {
            "variance_explained": float(self.variance_explained) if self.variance_explained else 0.0,
            "reconstruction_error": float(reconstruction_error),
            "similarity_correlation": float(similarity_correlation),
            "neighbor_preservation": neighbor_preservation,
            "compression_ratio": self.training_stats.get("compression_ratio", 0),
            "evaluation_samples": len(orig_sample)
        }
        
        logger.info(f"Quality evaluation completed:")
        logger.info(f"  Variance explained: {quality_metrics['variance_explained']:.1%}")
        logger.info(f"  Similarity correlation: {quality_metrics['similarity_correlation']:.3f}")
        logger.info(f"  Neighbor preservation: {quality_metrics['neighbor_preservation']:.1%}")
        
        return quality_metrics

    def _evaluate_neighbor_preservation(
        self,
        original_embeddings: np.ndarray,
        reduced_embeddings: np.ndarray,
        k: int = 10
    ) -> float:
        """Evaluate how well k-nearest neighbors are preserved."""
        if len(original_embeddings) < k + 1:
            return 1.0  # All neighbors preserved if too few samples
        
        preservation_scores = []
        
        for i in range(len(original_embeddings)):
            # Find k nearest neighbors in original space
            orig_similarities = cosine_similarity([original_embeddings[i]], original_embeddings)[0]
            orig_neighbors = np.argsort(orig_similarities)[::-1][1:k+1]  # Exclude self
            
            # Find k nearest neighbors in reduced space
            reduced_similarities = cosine_similarity([reduced_embeddings[i]], reduced_embeddings)[0]
            reduced_neighbors = np.argsort(reduced_similarities)[::-1][1:k+1]  # Exclude self
            
            # Calculate intersection
            intersection = len(set(orig_neighbors) & set(reduced_neighbors))
            preservation_scores.append(intersection / k)
        
        return np.mean(preservation_scores)

    def save_model(self, path: Optional[str] = None) -> str:
        """
        Save trained PCA model.
        
        Args:
            path: Optional path to save model
            
        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        save_path = path or self.model_path
        
        model_data = {
            "pca_model": self.pca_model,
            "scaler": self.scaler,
            "target_dimensions": self.target_dimensions,
            "preserve_variance": self.preserve_variance,
            "scale_features": self.scale_features,
            "variance_explained": self.variance_explained,
            "training_stats": self.training_stats,
            "is_fitted": self.is_fitted,
            "saved_at": datetime.now().isoformat()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save using joblib for sklearn compatibility
        joblib.dump(model_data, save_path)
        
        logger.info(f"PCA model saved to: {save_path}")
        return save_path

    def load_model(self, path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load trained PCA model.
        
        Args:
            path: Optional path to load model from
            
        Returns:
            Model metadata
        """
        load_path = path or self.model_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        try:
            model_data = joblib.load(load_path)
            
            self.pca_model = model_data["pca_model"]
            self.scaler = model_data["scaler"]
            self.target_dimensions = model_data["target_dimensions"]
            self.preserve_variance = model_data["preserve_variance"]
            self.scale_features = model_data["scale_features"]
            self.variance_explained = model_data["variance_explained"]
            self.training_stats = model_data["training_stats"]
            self.is_fitted = model_data["is_fitted"]
            
            logger.info(f"PCA model loaded from: {load_path}")
            logger.info(f"Model dimensions: {self.training_stats.get('original_dimensions')} → {self.training_stats.get('actual_dimensions')}")
            
            return {
                "loaded_from": load_path,
                "saved_at": model_data.get("saved_at"),
                "training_stats": self.training_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to load model from {load_path}: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.is_fitted:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "target_dimensions": self.target_dimensions,
            "actual_dimensions": self.pca_model.n_components_,
            "preserve_variance": self.preserve_variance,
            "variance_explained": float(self.variance_explained),
            "scale_features": self.scale_features,
            "training_stats": self.training_stats,
            "model_path": self.model_path
        }

    def batch_transform(
        self,
        embeddings_list: List[List[float]],
        batch_size: int = 1000
    ) -> List[List[float]]:
        """
        Transform embeddings in batches for memory efficiency.
        
        Args:
            embeddings_list: List of embedding vectors
            batch_size: Size of processing batches
            
        Returns:
            List of reduced embedding vectors
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transformation")
        
        logger.info(f"Batch transforming {len(embeddings_list)} embeddings in batches of {batch_size}")
        
        reduced_embeddings = []
        
        for i in range(0, len(embeddings_list), batch_size):
            batch = embeddings_list[i:i + batch_size]
            batch_array = np.array(batch)
            
            reduced_batch = self.transform(batch_array)
            reduced_embeddings.extend(reduced_batch.tolist())
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {i + len(batch)}/{len(embeddings_list)} embeddings")
        
        logger.info(f"Batch transformation completed: {len(reduced_embeddings)} embeddings")
        return reduced_embeddings

    @staticmethod
    def create_from_legacy_data(
        legacy_embeddings: Union[List[List[float]], np.ndarray],
        target_dimensions: int = 768,
        preserve_variance: float = 0.95,
        model_path: str = "models/legacy_pca_reducer.pkl"
    ) -> 'DimensionReducer':
        """
        Create and fit reducer from legacy embedding data.
        
        Args:
            legacy_embeddings: Legacy 3072-dimensional embeddings
            target_dimensions: Target dimensions for reduction
            preserve_variance: Minimum variance to preserve
            model_path: Path to save fitted model
            
        Returns:
            Fitted DimensionReducer instance
        """
        reducer = DimensionReducer(
            target_dimensions=target_dimensions,
            preserve_variance=preserve_variance,
            model_path=model_path
        )
        
        logger.info("Creating reducer from legacy data...")
        training_stats = reducer.fit(legacy_embeddings, save_model=True)
        
        logger.info("Legacy reducer created successfully:")
        logger.info(f"  Original dimensions: {training_stats['original_dimensions']}")
        logger.info(f"  Reduced dimensions: {training_stats['actual_dimensions']}")
        logger.info(f"  Variance explained: {training_stats['variance_explained']:.1%}")
        logger.info(f"  Compression ratio: {training_stats['compression_ratio']:.1f}x")
        
        return reducer