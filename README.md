# neural-consensus# Neural Consensus

**Neural Fault Detection with Transfer Learning for Distributed Consensus**

A research implementation exploring whether neural networks can detect and classify node failures faster and more accurately than traditional timeout-based methods in distributed consensus systems.

## 🎯 Research Question

> Can a neural network with transfer learning capabilities detect and classify node failures faster and more accurately than traditional timeout-based methods, while generalizing across different network deployments?

## 🔬 Key Innovations

1. **Predictive Failure Detection** — Detect failures *before* they happen using learned patterns
2. **Failure Classification** — Distinguish crash vs Byzantine vs partition vs slowdown
3. **Transfer Learning** — Train on one deployment, transfer to another with minimal fine-tuning

## 📁 Project Structure
```
neural-consensus/
├── simulation/              # Network simulation environment
│   ├── clock.py            # Discrete event simulation clock
│   ├── network.py          # Message passing with delays/loss/partitions
│   ├── node.py             # Base node with failure injection
│   └── failures.py         # Failure injection strategies
│
├── protocols/raft/         # Raft consensus implementation
│   ├── messages.py         # Raft message types (Vote, AppendEntries, etc.)
│   ├── state.py            # Raft state management
│   └── node.py             # Complete Raft node
│
├── neural/                 # Neural network components
│   ├── features.py         # Feature extraction from observations
│   ├── encoder.py          # LSTM autoencoder
│   ├── classifier.py       # Failure classification head
│   ├── detector.py         # Neural failure detector
│   ├── training.py         # Training loop
│   └── transfer.py         # Transfer learning utilities
│
├── data/                   # Data collection
│   ├── collector.py        # Observation collector
│   └── labeler.py          # Auto-labeling
│
├── experiments/            # Experiment scripts
├── configs/                # Configuration files
├── models/                 # Saved models
├── results/                # Experiment results
└── tests/                  # Unit tests
```

## 🚀 Quick Start

### Installation
```bash
# Clone and enter directory
cd neural-consensus

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_all.py
```

### Train the Neural Detector
```bash
python train_detector.py
```

### Run Experiments
```bash
python run_experiments.py
```

## 🧠 Neural Architecture
```
Input: [20 observations × 16 features]
              ↓
      ┌───────────────┐
      │ LSTM Encoder  │ (64 units, 2 layers)
      └───────────────┘
              ↓
      [32-dim latent space]
              ↓
     ┌────────┴────────┐
     ↓                 ↓
┌─────────┐    ┌──────────────┐
│ Decoder │    │  Classifier  │
└─────────┘    └──────────────┘
     ↓                 ↓
Reconstruction    Failure Type
   Error          Prediction
(anomaly score)
```

### Failure Classes

| Class | Description |
|-------|-------------|
| 0 - Healthy | Normal operation |
| 1 - Pre-failure | About to fail (within 5s) |
| 2 - Crashed | Node has stopped |
| 3 - Byzantine | Malicious behavior |
| 4 - Partitioned | Network split |
| 5 - Slow | Degraded performance |

### Features (16 per observation)

- Latency: mean, std, trend, jitter
- Messages: rate, drop rate
- Heartbeats: regularity, missed count
- Response: rate, time
- Raft: term freshness, log/commit progress, leader status
- Composite: health score

## 📊 Experiments

### 1. Detection Speed
Compare time-to-detection between neural and timeout-based approaches.

### 2. False Positive Rate
Measure false alarms under various network conditions.

### 3. Classification Accuracy
Evaluate failure type classification with confusion matrix.

### 4. Transfer Learning
Test model transfer across different network deployments.

### 5. End-to-End Performance
Measure impact on consensus throughput, latency, and availability.

## 🔧 Configuration

See `configs/default.yaml` for all options:
```yaml
neural_detector:
  window_size: 20
  encoder:
    hidden_size: 64
    latent_size: 32
  classifier:
    hidden_sizes: [64, 32]
    num_classes: 6
  training:
    epochs: 100
    learning_rate: 0.001
```

## 📈 Results

After training, results are saved to `results/`:
- `training_history.png` — Loss curves
- `confusion_matrix.png` — Classification performance
- `detection_latency.png` — Time to detect failures
- `transfer_performance.png` — Transfer learning results

## 🔮 Blockchain Applications

This research directly applies to:
- **Proof of Stake** validator monitoring (Ethereum, Solana)
- **BFT chains** (Cosmos/Tendermint, BNB Chain)
- **Layer 2** sequencer monitoring
- **Cross-chain bridges** validator security

## 📚 References

1. Ongaro & Ousterhout. "In Search of an Understandable Consensus Algorithm" (Raft)
2. Castro & Liskov. "Practical Byzantine Fault Tolerance"
3. Kleppmann. "Designing Data-Intensive Applications"
4. Chandra & Toueg. "Unreliable Failure Detectors for Reliable Distributed Systems"

## 📄 License

MIT License

## 📖 Citation
```bibtex
@article{neural-consensus-2025,
  title={Neural Fault Detection with Transfer Learning for Distributed Consensus},
  author={Prakhar},
  year={2025}
}
``` 