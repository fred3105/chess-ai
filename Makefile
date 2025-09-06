.PHONY: dataset train all clean help

# Default chunk size (can be overridden: make dataset CHUNK_SIZE=2000000)
CHUNK_SIZE ?= 1000000
DATA_DIR ?= data/
DATASET_DIR ?= chunked_dataset/
WANDB_PROJECT ?= chess-eval-optimization

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

dataset: ## Create chunked dataset from PGN files
	@echo "🚀 Creating chunked dataset..."
	@echo "📊 Chunk size: $(CHUNK_SIZE)"
	@echo "📁 Data directory: $(DATA_DIR)"
	@echo "💾 Output directory: $(DATASET_DIR)"
	uv run training/create_chunked_dataset.py --data-dir $(DATA_DIR) --chunk-size $(CHUNK_SIZE) --output-dir $(DATASET_DIR)

train: ## Train NNUE model on chunked dataset
	@echo "🏋️ Starting NNUE training..."
	@echo "📦 Dataset directory: $(DATASET_DIR)"
	@echo "📊 Wandb project: $(WANDB_PROJECT)"
	uv run training/prepare_and_train.py --chunked-training --positions-file $(DATASET_DIR) --wandb-project $(WANDB_PROJECT)

all: dataset train ## Create dataset and train model (full pipeline)

clean: ## Clean up generated files
	@echo "🧹 Cleaning up..."
	rm -rf $(DATASET_DIR)
	rm -f training.log dataset_creation.log
	rm -rf checkpoints/
	rm -rf wandb/

# Quick targets for different chunk sizes
dataset-small: ## Create dataset with 10K chunks (for testing)
	$(MAKE) dataset CHUNK_SIZE=50000

dataset-medium: ## Create dataset with 500K chunks
	$(MAKE) dataset CHUNK_SIZE=500000

dataset-large: ## Create dataset with 2M chunks
	$(MAKE) dataset CHUNK_SIZE=2000000

# Training with specific configurations
train-fast: ## Train with minimal settings for quick testing
	uv run training/prepare_and_train.py --chunked-training --positions-file $(DATASET_DIR) --epochs 1 --wandb-project $(WANDB_PROJECT)-fast

train-full: ## Train with full 200 epochs for maximum strength
	uv run training/prepare_and_train.py --chunked-training --positions-file $(DATASET_DIR) --epochs 200 --wandb-project $(WANDB_PROJECT)-full

# Development targets
test-dataset: dataset-small ## Create small test dataset (100K chunks)
	@echo "✅ Small test dataset created"

# Pipeline targets
quick-pipeline: dataset-small train-fast ## Quick test pipeline (small dataset + short training)

full-pipeline: dataset train-full ## Full production pipeline (1M chunks + 200 epochs)