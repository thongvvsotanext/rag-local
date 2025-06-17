import faiss
import numpy as np
import pickle
import os
import logging
import time
import traceback
import json
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class M2OptimizedFAISSStore:
    def __init__(self, dimension=384, index_type="Flat"):  # Updated default to BGE-small's actual dimension
        """Initialize FAISS store optimized for M2 Mac.
        
        Args:
            dimension (int): Embedding dimension (384 for BGE-small, 768 for BGE-base, 1024 for BGE-large)
            index_type (str): Type of FAISS index to use ("Flat", "IVFFlat", "HNSW")
        """
        self.dimension = dimension
        self.index_type = index_type
        self.document_index = None
        self.chat_index = None
        self.document_id_map = []  # Maps FAISS indices to chunk_ids
        self.chat_id_map = []      # Maps FAISS indices to message_ids
        self.stats = {
            'total_document_additions': 0,
            'total_chat_additions': 0,
            'total_document_searches': 0,
            'total_chat_searches': 0,
            'last_save_time': None,
            'last_load_time': None,
            'initialization_time': time.time()
        }
        
        logger.info(f"FAISS_INIT - Initializing FAISS store", extra={
            'dimension': dimension,
            'index_type': index_type,
            'timestamp': time.time()
        })
        
        start_time = time.time()
        
        # Initialize indices optimized for M2 Mac
        try:
            self._init_document_index()
            self._init_chat_index()
            
            init_time = time.time() - start_time
            self.stats['initialization_time'] = init_time
            
            logger.info(f"FAISS_INIT_SUCCESS - FAISS store initialized successfully", extra={
                'dimension': dimension,
                'index_type': index_type,
                'initialization_time_seconds': round(init_time, 3),
                'document_index_type': type(self.document_index).__name__,
                'chat_index_type': type(self.chat_index).__name__
            })
        except Exception as e:
            logger.error(f"FAISS_INIT_ERROR - Failed to initialize FAISS store", extra={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'dimension': dimension,
                'index_type': index_type
            })
            raise
    
    def _validate_dimensions(self, embeddings, context="embeddings"):
        """Validate embedding dimensions match expected dimension."""
        start_time = time.time()
        
        try:
            embeddings_array = np.array(embeddings)
            if embeddings_array.ndim == 1:
                embeddings_array = embeddings_array.reshape(1, -1)
            
            if embeddings_array.shape[1] != self.dimension:
                error_msg = (
                    f"Dimension mismatch in {context}: expected {self.dimension}, "
                    f"got {embeddings_array.shape[1]}. Shape: {embeddings_array.shape}"
                )
                logger.error(f"FAISS_VALIDATION_ERROR - {error_msg}", extra={
                    'context': context,
                    'expected_dimension': self.dimension,
                    'actual_dimension': embeddings_array.shape[1],
                    'embedding_shape': embeddings_array.shape
                })
                raise ValueError(error_msg)
            
            validation_time = time.time() - start_time
            logger.debug(f"FAISS_VALIDATION_SUCCESS - Dimension validation passed", extra={
                'context': context,
                'dimension': self.dimension,
                'embedding_count': embeddings_array.shape[0],
                'validation_time_seconds': round(validation_time, 4)
            })
            
            return embeddings_array.astype('float32')
            
        except Exception as e:
            logger.error(f"FAISS_VALIDATION_ERROR - Validation failed for {context}", extra={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'context': context,
                'expected_dimension': self.dimension
            })
            raise
    
    def _init_document_index(self):
        """Initialize FAISS index optimized for M2 Mac"""
        start_time = time.time()
        
        try:
            logger.info(f"FAISS_INDEX_INIT - Initializing document index", extra={
                'index_type': self.index_type,
                'dimension': self.dimension
            })
            
            if self.index_type == "IVFFlat":
                # For M2 Mac: Smaller clusters to save memory
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.document_index = faiss.IndexIVFFlat(quantizer, self.dimension, 50)  # Reduced from 100
                # Set search parameters for better performance on M2
                self.document_index.nprobe = 5  # Reduced probe count
                
                logger.info(f"FAISS_INDEX_INIT - IVFFlat index created", extra={
                    'clusters': 50,
                    'nprobe': 5,
                    'dimension': self.dimension
                })
                
            elif self.index_type == "HNSW":
                # Conservative HNSW settings for M2 Mac
                self.document_index = faiss.IndexHNSWFlat(self.dimension, 16)  # Reduced M parameter
                self.document_index.hnsw.efConstruction = 32  # Reduced construction
                self.document_index.hnsw.efSearch = 16        # Reduced search
                
                logger.info(f"FAISS_INDEX_INIT - HNSW index created", extra={
                    'M': 16,
                    'efConstruction': 32,
                    'efSearch': 16,
                    'dimension': self.dimension
                })
            else:
                # Default: Simple flat index - most memory efficient
                self.document_index = faiss.IndexFlatIP(self.dimension)
                
                logger.info(f"FAISS_INDEX_INIT - Flat index created", extra={
                    'dimension': self.dimension,
                    'note': 'Most memory efficient option'
                })
            
            init_time = time.time() - start_time
            logger.info(f"FAISS_INDEX_INIT_SUCCESS - Document index initialized", extra={
                'index_type': type(self.document_index).__name__,
                'dimension': self.dimension,
                'initialization_time_seconds': round(init_time, 3),
                'is_trained': getattr(self.document_index, 'is_trained', True)
            })
            
        except Exception as e:
            logger.error(f"FAISS_INDEX_INIT_ERROR - Failed to initialize document index", extra={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'index_type': self.index_type,
                'dimension': self.dimension
            })
            raise
    
    def _init_chat_index(self):
        """Initialize FAISS index for chat messages - always simple for M2"""
        start_time = time.time()
        
        try:
            logger.info(f"FAISS_CHAT_INDEX_INIT - Initializing chat index", extra={
                'dimension': self.dimension
            })
            
            self.chat_index = faiss.IndexFlatIP(self.dimension)
            
            init_time = time.time() - start_time
            logger.info(f"FAISS_CHAT_INDEX_INIT_SUCCESS - Chat index initialized", extra={
                'index_type': type(self.chat_index).__name__,
                'dimension': self.dimension,
                'initialization_time_seconds': round(init_time, 3)
            })
            
        except Exception as e:
            logger.error(f"FAISS_CHAT_INDEX_INIT_ERROR - Failed to initialize chat index", extra={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'dimension': self.dimension
            })
            raise
    
    def add_documents(self, embeddings, chunk_ids, batch_size=32):
        """Add document embeddings in smaller batches for M2 Mac"""
        operation_start_time = time.time()
        
        try:
            logger.info(f"FAISS_ADD_DOCS_START - Starting document addition", extra={
                'embedding_count': len(embeddings),
                'chunk_ids_count': len(chunk_ids),
                'batch_size': batch_size,
                'current_index_size': self.document_index.ntotal if self.document_index else 0
            })
            
            if len(embeddings) != len(chunk_ids):
                error_msg = f"Mismatch between embeddings count ({len(embeddings)}) and chunk_ids count ({len(chunk_ids)})"
                logger.error(f"FAISS_ADD_DOCS_ERROR - {error_msg}")
                raise ValueError(error_msg)
            
            # Validate dimensions
            validation_start_time = time.time()
            embeddings = self._validate_dimensions(embeddings, "document embeddings")
            validation_time = time.time() - validation_start_time
            
            logger.info(f"FAISS_ADD_DOCS_STEP - Validation completed", extra={
                'embedding_count': len(embeddings),
                'validation_time_seconds': round(validation_time, 3),
                'embedding_shape': embeddings.shape
            })
            
            # Process in smaller batches to manage memory
            batches_processed = 0
            total_batches = (len(embeddings) + batch_size - 1) // batch_size
            
            for i in range(0, len(embeddings), batch_size):
                batch_start_time = time.time()
                batch_embeddings = embeddings[i:i + batch_size]
                batch_ids = chunk_ids[i:i + batch_size]
                
                logger.debug(f"FAISS_ADD_DOCS_BATCH - Processing batch {batches_processed + 1}/{total_batches}", extra={
                    'batch_start_index': i,
                    'batch_size': len(batch_embeddings),
                    'batch_ids_sample': batch_ids[:3] if len(batch_ids) > 3 else batch_ids
                })
                
                # Normalize embeddings for cosine similarity
                normalize_start_time = time.time()
                faiss.normalize_L2(batch_embeddings)
                normalize_time = time.time() - normalize_start_time
                
                # Train index if needed (for IVF indices) with smaller training set
                if isinstance(self.document_index, faiss.IndexIVFFlat) and not self.document_index.is_trained:
                    if len(batch_embeddings) >= 50:  # Reduced training requirement
                        train_start_time = time.time()
                        logger.info(f"FAISS_ADD_DOCS_TRAINING - Training IVF index", extra={
                            'training_samples': len(batch_embeddings)
                        })
                        self.document_index.train(batch_embeddings)
                        train_time = time.time() - train_start_time
                        
                        logger.info(f"FAISS_ADD_DOCS_TRAINING_SUCCESS - IVF index trained", extra={
                            'training_time_seconds': round(train_time, 3),
                            'training_samples': len(batch_embeddings),
                            'is_trained': self.document_index.is_trained
                        })
                
                # Add embeddings
                add_start_time = time.time()
                initial_count = self.document_index.ntotal
                self.document_index.add(batch_embeddings)
                final_count = self.document_index.ntotal
                add_time = time.time() - add_start_time
                
                # Update ID mapping
                self.document_id_map.extend(batch_ids)
                
                batch_time = time.time() - batch_start_time
                batches_processed += 1
                
                logger.debug(f"FAISS_ADD_DOCS_BATCH_SUCCESS - Batch processed", extra={
                    'batch_number': batches_processed,
                    'total_batches': total_batches,
                    'batch_size': len(batch_embeddings),
                    'normalize_time_seconds': round(normalize_time, 4),
                    'add_time_seconds': round(add_time, 4),
                    'total_batch_time_seconds': round(batch_time, 3),
                    'index_size_before': initial_count,
                    'index_size_after': final_count,
                    'vectors_added': final_count - initial_count
                })
            
            operation_time = time.time() - operation_start_time
            self.stats['total_document_additions'] += len(embeddings)
            
            logger.info(f"FAISS_ADD_DOCS_SUCCESS - Document addition completed", extra={
                'total_embeddings_added': len(embeddings),
                'total_batches_processed': batches_processed,
                'final_index_size': self.document_index.ntotal,
                'total_id_mappings': len(self.document_id_map),
                'operation_time_seconds': round(operation_time, 3),
                'avg_time_per_embedding_ms': round((operation_time / len(embeddings)) * 1000, 2),
                'cumulative_document_additions': self.stats['total_document_additions']
            })
            
            return len(self.document_id_map) - len(chunk_ids)
            
        except Exception as e:
            operation_time = time.time() - operation_start_time
            logger.error(f"FAISS_ADD_DOCS_ERROR - Document addition failed", extra={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'embedding_count': len(embeddings),
                'chunk_ids_count': len(chunk_ids),
                'batch_size': batch_size,
                'operation_time_seconds': round(operation_time, 3),
                'current_index_size': self.document_index.ntotal if self.document_index else 0
            })
            raise
    
    def add_chat_messages(self, embeddings, message_ids):
        """Add chat message embeddings - optimized for M2"""
        operation_start_time = time.time()
        
        try:
            logger.info(f"FAISS_ADD_CHAT_START - Starting chat message addition", extra={
                'embedding_count': len(embeddings),
                'message_ids_count': len(message_ids),
                'current_index_size': self.chat_index.ntotal if self.chat_index else 0
            })
            
            if len(embeddings) != len(message_ids):
                error_msg = f"Mismatch between embeddings count ({len(embeddings)}) and message_ids count ({len(message_ids)})"
                logger.error(f"FAISS_ADD_CHAT_ERROR - {error_msg}")
                raise ValueError(error_msg)
            
            # Validate dimensions
            validation_start_time = time.time()
            embeddings = self._validate_dimensions(embeddings, "chat message embeddings")
            validation_time = time.time() - validation_start_time
            
            # Normalize embeddings
            normalize_start_time = time.time()
            faiss.normalize_L2(embeddings)
            normalize_time = time.time() - normalize_start_time
            
            # Add to index
            add_start_time = time.time()
            start_idx = self.chat_index.ntotal
            initial_count = self.chat_index.ntotal
            self.chat_index.add(embeddings)
            final_count = self.chat_index.ntotal
            add_time = time.time() - add_start_time
            
            # Update ID mapping
            self.chat_id_map.extend(message_ids)
            
            operation_time = time.time() - operation_start_time
            self.stats['total_chat_additions'] += len(embeddings)
            
            logger.info(f"FAISS_ADD_CHAT_SUCCESS - Chat message addition completed", extra={
                'embeddings_added': len(embeddings),
                'start_index': start_idx,
                'final_index_size': final_count,
                'vectors_added': final_count - initial_count,
                'total_id_mappings': len(self.chat_id_map),
                'validation_time_seconds': round(validation_time, 4),
                'normalize_time_seconds': round(normalize_time, 4),
                'add_time_seconds': round(add_time, 4),
                'total_operation_time_seconds': round(operation_time, 3),
                'cumulative_chat_additions': self.stats['total_chat_additions']
            })
            
            return start_idx
            
        except Exception as e:
            operation_time = time.time() - operation_start_time
            logger.error(f"FAISS_ADD_CHAT_ERROR - Chat message addition failed", extra={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'embedding_count': len(embeddings),
                'message_ids_count': len(message_ids),
                'operation_time_seconds': round(operation_time, 3),
                'current_index_size': self.chat_index.ntotal if self.chat_index else 0
            })
            raise
    
    def search_documents(self, query_embedding, top_k=5, threshold=0.75):
        """Search with M2-optimized parameters"""
        search_start_time = time.time()
        
        try:
            logger.debug(f"FAISS_SEARCH_DOCS_START - Starting document search", extra={
                'top_k': top_k,
                'threshold': threshold,
                'index_size': self.document_index.ntotal if self.document_index else 0,
                'id_map_size': len(self.document_id_map)
            })
            
            # Validate query embedding
            validation_start_time = time.time()
            query_embedding = self._validate_dimensions([query_embedding], "query embedding")
            validation_time = time.time() - validation_start_time
            
            # Normalize query
            normalize_start_time = time.time()
            faiss.normalize_L2(query_embedding)
            normalize_time = time.time() - normalize_start_time
            
            # Check if index has any documents
            if self.document_index.ntotal == 0:
                logger.warning(f"FAISS_SEARCH_DOCS_WARNING - Document index is empty", extra={
                    'index_size': 0,
                    'search_time_seconds': round(time.time() - search_start_time, 3)
                })
                return []
            
            # Adjust search parameters for M2 performance
            original_nprobe = None
            if isinstance(self.document_index, faiss.IndexIVFFlat):
                original_nprobe = self.document_index.nprobe
                self.document_index.nprobe = min(5, original_nprobe)
                logger.debug(f"FAISS_SEARCH_DOCS_CONFIG - Adjusted nprobe for M2", extra={
                    'original_nprobe': original_nprobe,
                    'adjusted_nprobe': self.document_index.nprobe
                })
            
            # Perform search
            faiss_search_start_time = time.time()
            scores, indices = self.document_index.search(query_embedding, top_k)
            faiss_search_time = time.time() - faiss_search_start_time
            
            # Process results
            process_start_time = time.time()
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx < len(self.document_id_map) and score >= threshold:
                    results.append({
                        'chunk_id': self.document_id_map[idx],
                        'score': float(score),
                        'faiss_index': int(idx)
                    })
            process_time = time.time() - process_start_time
            
            search_time = time.time() - search_start_time
            self.stats['total_document_searches'] += 1
            
            logger.debug(f"FAISS_SEARCH_DOCS_SUCCESS - Document search completed", extra={
                'top_k_requested': top_k,
                'results_returned': len(results),
                'threshold': threshold,
                'index_size': self.document_index.ntotal,
                'validation_time_seconds': round(validation_time, 4),
                'normalize_time_seconds': round(normalize_time, 4),
                'faiss_search_time_seconds': round(faiss_search_time, 4),
                'process_time_seconds': round(process_time, 4),
                'total_search_time_seconds': round(search_time, 3),
                'scores_range': {
                    'max': float(np.max(scores[0])) if len(scores[0]) > 0 else None,
                    'min': float(np.min(scores[0])) if len(scores[0]) > 0 else None
                },
                'cumulative_document_searches': self.stats['total_document_searches']
            })
            
            return results
            
        except Exception as e:
            search_time = time.time() - search_start_time
            logger.error(f"FAISS_SEARCH_DOCS_ERROR - Document search failed", extra={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'top_k': top_k,
                'threshold': threshold,
                'search_time_seconds': round(search_time, 3),
                'index_size': self.document_index.ntotal if self.document_index else 0
            })
            raise
    
    def search_chat_messages(self, query_embedding, top_k=3, threshold=0.7):
        """Search chat messages with M2 optimization"""
        search_start_time = time.time()
        
        try:
            logger.debug(f"FAISS_SEARCH_CHAT_START - Starting chat message search", extra={
                'top_k': top_k,
                'threshold': threshold,
                'index_size': self.chat_index.ntotal if self.chat_index else 0,
                'id_map_size': len(self.chat_id_map)
            })
            
            # Validate query embedding
            validation_start_time = time.time()
            query_embedding = self._validate_dimensions([query_embedding], "query embedding")
            validation_time = time.time() - validation_start_time
            
            # Normalize query
            normalize_start_time = time.time()
            faiss.normalize_L2(query_embedding)
            normalize_time = time.time() - normalize_start_time
            
            # Check if index has any messages
            if self.chat_index.ntotal == 0:
                logger.warning(f"FAISS_SEARCH_CHAT_WARNING - Chat index is empty", extra={
                    'index_size': 0,
                    'search_time_seconds': round(time.time() - search_start_time, 3)
                })
                return []
            
            # Perform search
            faiss_search_start_time = time.time()
            scores, indices = self.chat_index.search(query_embedding, top_k)
            faiss_search_time = time.time() - faiss_search_start_time
            
            # Process results
            process_start_time = time.time()
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx < len(self.chat_id_map) and score >= threshold:
                    results.append({
                        'message_id': self.chat_id_map[idx],
                        'score': float(score),
                        'faiss_index': int(idx)
                    })
            process_time = time.time() - process_start_time
            
            search_time = time.time() - search_start_time
            self.stats['total_chat_searches'] += 1
            
            logger.debug(f"FAISS_SEARCH_CHAT_SUCCESS - Chat message search completed", extra={
                'top_k_requested': top_k,
                'results_returned': len(results),
                'threshold': threshold,
                'index_size': self.chat_index.ntotal,
                'validation_time_seconds': round(validation_time, 4),
                'normalize_time_seconds': round(normalize_time, 4),
                'faiss_search_time_seconds': round(faiss_search_time, 4),
                'process_time_seconds': round(process_time, 4),
                'total_search_time_seconds': round(search_time, 3),
                'scores_range': {
                    'max': float(np.max(scores[0])) if len(scores[0]) > 0 else None,
                    'min': float(np.min(scores[0])) if len(scores[0]) > 0 else None
                },
                'cumulative_chat_searches': self.stats['total_chat_searches']
            })
            
            return results
            
        except Exception as e:
            search_time = time.time() - search_start_time
            logger.error(f"FAISS_SEARCH_CHAT_ERROR - Chat message search failed", extra={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'top_k': top_k,
                'threshold': threshold,
                'search_time_seconds': round(search_time, 3),
                'index_size': self.chat_index.ntotal if self.chat_index else 0
            })
            raise
    
    def save_indices(self, base_path="/app/data/faiss"):
        """Save FAISS indices with compression for M2 storage"""
        save_start_time = time.time()
        
        try:
            logger.info(f"FAISS_SAVE_START - Starting index save operation", extra={
                'base_path': base_path,
                'document_index_size': self.document_index.ntotal if self.document_index else 0,
                'chat_index_size': self.chat_index.ntotal if self.chat_index else 0,
                'document_id_map_size': len(self.document_id_map),
                'chat_id_map_size': len(self.chat_id_map)
            })
            
            # Create directory
            os.makedirs(base_path, exist_ok=True)
            
            # Save FAISS indices
            doc_save_start = time.time()
            faiss.write_index(self.document_index, f"{base_path}/document_index.faiss")
            doc_save_time = time.time() - doc_save_start
            
            chat_save_start = time.time()
            faiss.write_index(self.chat_index, f"{base_path}/chat_index.faiss")
            chat_save_time = time.time() - chat_save_start
            
            # Save ID mappings with compression
            doc_map_save_start = time.time()
            with open(f"{base_path}/document_id_map.pkl", 'wb') as f:
                pickle.dump(self.document_id_map, f, protocol=pickle.HIGHEST_PROTOCOL)
            doc_map_save_time = time.time() - doc_map_save_start
            
            chat_map_save_start = time.time()
            with open(f"{base_path}/chat_id_map.pkl", 'wb') as f:
                pickle.dump(self.chat_id_map, f, protocol=pickle.HIGHEST_PROTOCOL)
            chat_map_save_time = time.time() - chat_map_save_start
            
            # Save metadata for validation
            metadata_save_start = time.time()
            metadata = {
                'dimension': self.dimension,
                'index_type': self.index_type,
                'document_count': self.document_index.ntotal,
                'chat_count': self.chat_index.ntotal,
                'document_id_map_size': len(self.document_id_map),
                'chat_id_map_size': len(self.chat_id_map),
                'save_timestamp': time.time(),
                'stats': self.stats.copy()
            }
            with open(f"{base_path}/metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
            metadata_save_time = time.time() - metadata_save_start
            
            # Get file sizes for logging
            try:
                doc_index_size = os.path.getsize(f"{base_path}/document_index.faiss")
                chat_index_size = os.path.getsize(f"{base_path}/chat_index.faiss")
                doc_map_size = os.path.getsize(f"{base_path}/document_id_map.pkl")
                chat_map_size = os.path.getsize(f"{base_path}/chat_id_map.pkl")
                metadata_size = os.path.getsize(f"{base_path}/metadata.pkl")
                total_size = doc_index_size + chat_index_size + doc_map_size + chat_map_size + metadata_size
            except OSError:
                doc_index_size = chat_index_size = doc_map_size = chat_map_size = metadata_size = total_size = 0
            
            save_time = time.time() - save_start_time
            self.stats['last_save_time'] = time.time()
            
            logger.info(f"FAISS_SAVE_SUCCESS - Index save operation completed", extra={
                'base_path': base_path,
                'document_vectors_saved': self.document_index.ntotal,
                'chat_vectors_saved': self.chat_index.ntotal,
                'total_save_time_seconds': round(save_time, 3),
                'timing_breakdown': {
                    'document_index_save_seconds': round(doc_save_time, 3),
                    'chat_index_save_seconds': round(chat_save_time, 3),
                    'document_map_save_seconds': round(doc_map_save_time, 3),
                    'chat_map_save_seconds': round(chat_map_save_time, 3),
                    'metadata_save_seconds': round(metadata_save_time, 3)
                },
                'file_sizes': {
                    'document_index_mb': round(doc_index_size / (1024 * 1024), 2),
                    'chat_index_mb': round(chat_index_size / (1024 * 1024), 2),
                    'document_map_mb': round(doc_map_size / (1024 * 1024), 2),
                    'chat_map_mb': round(chat_map_size / (1024 * 1024), 2),
                    'metadata_mb': round(metadata_size / (1024 * 1024), 2),
                    'total_mb': round(total_size / (1024 * 1024), 2)
                }
            })
            
        except Exception as e:
            save_time = time.time() - save_start_time
            logger.error(f"FAISS_SAVE_ERROR - Index save operation failed", extra={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'base_path': base_path,
                'save_time_seconds': round(save_time, 3),
                'document_index_size': self.document_index.ntotal if self.document_index else 0,
                'chat_index_size': self.chat_index.ntotal if self.chat_index else 0
            })
            raise
    
    def load_indices(self, base_path="/app/data/faiss"):
        """Load FAISS indices with error handling for M2"""
        load_start_time = time.time()
        
        try:
            logger.info(f"FAISS_LOAD_START - Starting index load operation", extra={
                'base_path': base_path,
                'expected_dimension': self.dimension
            })
            
            if not os.path.exists(f"{base_path}/document_index.faiss"):
                logger.info(f"FAISS_LOAD_INFO - No existing indices found, starting fresh", extra={
                    'base_path': base_path,
                    'load_time_seconds': round(time.time() - load_start_time, 3)
                })
                return False
            
            # Load and validate metadata
            metadata_load_start = time.time()
            metadata = None
            if os.path.exists(f"{base_path}/metadata.pkl"):
                try:
                    with open(f"{base_path}/metadata.pkl", 'rb') as f:
                        metadata = pickle.load(f)
                    metadata_load_time = time.time() - metadata_load_start
                    
                    logger.info(f"FAISS_LOAD_METADATA - Metadata loaded", extra={
                        'metadata': metadata,
                        'metadata_load_time_seconds': round(metadata_load_time, 3)
                    })
                    
                    if metadata.get('dimension') != self.dimension:
                        logger.warning(f"FAISS_LOAD_WARNING - Dimension mismatch in saved indices", extra={
                            'expected_dimension': self.dimension,
                            'saved_dimension': metadata.get('dimension'),
                            'reinitializing': True
                        })
                        self._init_document_index()
                        self._init_chat_index()
                        self.document_id_map = []
                        self.chat_id_map = []
                        return False
                except Exception as e:
                    logger.warning(f"FAISS_LOAD_WARNING - Failed to load metadata, proceeding anyway", extra={
                        'error': str(e),
                        'metadata_load_time_seconds': round(time.time() - metadata_load_start, 3)
                    })
            
            # Load FAISS indices
            doc_load_start = time.time()
            self.document_index = faiss.read_index(f"{base_path}/document_index.faiss")
            doc_load_time = time.time() - doc_load_start
            
            chat_load_start = time.time()
            self.chat_index = faiss.read_index(f"{base_path}/chat_index.faiss")
            chat_load_time = time.time() - chat_load_start
            
            # Validate loaded indices dimensions
            if self.document_index.d != self.dimension:
                error_msg = f"Document index dimension mismatch: expected {self.dimension}, got {self.document_index.d}"
                logger.error(f"FAISS_LOAD_ERROR - {error_msg}")
                raise ValueError(error_msg)
            if self.chat_index.d != self.dimension:
                error_msg = f"Chat index dimension mismatch: expected {self.dimension}, got {self.chat_index.d}"
                logger.error(f"FAISS_LOAD_ERROR - {error_msg}")
                raise ValueError(error_msg)
            
            # Load ID mappings
            doc_map_load_start = time.time()
            with open(f"{base_path}/document_id_map.pkl", 'rb') as f:
                self.document_id_map = pickle.load(f)
            doc_map_load_time = time.time() - doc_map_load_start
            
            chat_map_load_start = time.time()
            with open(f"{base_path}/chat_id_map.pkl", 'rb') as f:
                self.chat_id_map = pickle.load(f)
            chat_map_load_time = time.time() - chat_map_load_start
            
            # Validate consistency
            if len(self.document_id_map) != self.document_index.ntotal:
                logger.warning(f"FAISS_LOAD_WARNING - Document ID map size mismatch", extra={
                    'id_map_size': len(self.document_id_map),
                    'index_size': self.document_index.ntotal
                })
            
            if len(self.chat_id_map) != self.chat_index.ntotal:
                logger.warning(f"FAISS_LOAD_WARNING - Chat ID map size mismatch", extra={
                    'id_map_size': len(self.chat_id_map),
                    'index_size': self.chat_index.ntotal
                })
            
            load_time = time.time() - load_start_time
            self.stats['last_load_time'] = time.time()
            
            logger.info(f"FAISS_LOAD_SUCCESS - Index load operation completed", extra={
                'base_path': base_path,
                'document_vectors_loaded': self.document_index.ntotal,
                'chat_vectors_loaded': self.chat_index.ntotal,
                'document_id_map_size': len(self.document_id_map),
                'chat_id_map_size': len(self.chat_id_map),
                'total_load_time_seconds': round(load_time, 3),
                'timing_breakdown': {
                    'metadata_load_seconds': round(metadata_load_time if metadata else 0, 3),
                    'document_index_load_seconds': round(doc_load_time, 3),
                    'chat_index_load_seconds': round(chat_load_time, 3),
                    'document_map_load_seconds': round(doc_map_load_time, 3),
                    'chat_map_load_seconds': round(chat_map_load_time, 3)
                },
                'dimension_validation': {
                    'expected': self.dimension,
                    'document_index': self.document_index.d,
                    'chat_index': self.chat_index.d,
                    'valid': True
                },
                'loaded_metadata': metadata
            })
            
            return True
            
        except Exception as e:
            load_time = time.time() - load_start_time
            logger.error(f"FAISS_LOAD_ERROR - Index load operation failed", extra={
                'error': str(e),
                'traceback': traceback.format_exc(),
                'base_path': base_path,
                'load_time_seconds': round(load_time, 3)
            })
            
            # Reinitialize on error
            logger.info(f"FAISS_LOAD_RECOVERY - Reinitializing indices after load failure")
            self._init_document_index()
            self._init_chat_index()
            self.document_id_map = []
            self.chat_id_map = []
            return False
    
    def get_memory_usage(self):
        """Get memory usage statistics for monitoring on M2"""
        try:
            doc_size = self.document_index.ntotal * self.dimension * 4  # float32
            chat_size = self.chat_index.ntotal * self.dimension * 4
            id_map_size = (len(self.document_id_map) + len(self.chat_id_map)) * 50  # Estimate for string IDs
            
            memory_stats = {
                "document_vectors": self.document_index.ntotal,
                "chat_vectors": self.chat_index.ntotal,
                "dimension": self.dimension,
                "estimated_memory_mb": round((doc_size + chat_size + id_map_size) / (1024 * 1024), 2),
                "breakdown": {
                    "document_embeddings_mb": round(doc_size / (1024 * 1024), 2),
                    "chat_embeddings_mb": round(chat_size / (1024 * 1024), 2),
                    "id_mappings_mb": round(id_map_size / (1024 * 1024), 2)
                },
                "document_index_type": type(self.document_index).__name__,
                "chat_index_type": type(self.chat_index).__name__,
                "stats": self.stats.copy(),
                "uptime_seconds": round(time.time() - self.stats['initialization_time'], 1)
            }
            
            logger.debug(f"FAISS_MEMORY_STATS - Memory usage calculated", extra=memory_stats)
            return memory_stats
            
        except Exception as e:
            logger.error(f"FAISS_MEMORY_STATS_ERROR - Failed to calculate memory usage", extra={
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return {
                "error": str(e),
                "document_vectors": getattr(self.document_index, 'ntotal', 0),
                "chat_vectors": getattr(self.chat_index, 'ntotal', 0),
                "dimension": self.dimension
            }
    
    def get_stats(self):
        """Get comprehensive statistics about the FAISS store."""
        try:
            current_time = time.time()
            uptime = current_time - self.stats['initialization_time']
            
            comprehensive_stats = {
                "indices": {
                    "document_index": {
                        "type": type(self.document_index).__name__,
                        "vector_count": self.document_index.ntotal,
                        "dimension": self.document_index.d,
                        "is_trained": getattr(self.document_index, 'is_trained', True)
                    },
                    "chat_index": {
                        "type": type(self.chat_index).__name__,
                        "vector_count": self.chat_index.ntotal,
                        "dimension": self.chat_index.d,
                        "is_trained": getattr(self.chat_index, 'is_trained', True)
                    }
                },
                "id_mappings": {
                    "document_mappings": len(self.document_id_map),
                    "chat_mappings": len(self.chat_id_map)
                },
                "operations": self.stats.copy(),
                "performance": {
                    "uptime_seconds": round(uptime, 1),
                    "avg_docs_per_addition": round(self.stats['total_document_additions'] / max(1, self.stats.get('addition_operations', 1)), 2),
                    "searches_per_second": round(self.stats['total_document_searches'] / max(uptime, 1), 3)
                },
                "memory": self.get_memory_usage()
            }
            
            logger.debug(f"FAISS_COMPREHENSIVE_STATS - Statistics compiled", extra=comprehensive_stats)
            return comprehensive_stats
            
        except Exception as e:
            logger.error(f"FAISS_STATS_ERROR - Failed to compile statistics", extra={
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return {"error": str(e), "basic_stats": self.stats.copy()}