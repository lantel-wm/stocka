# Stocka - A-Share Quantitative Backtesting Framework

A simple, accurate, and extensible daily backtesting framework for Chinese A-share market.

## Features

- **Accuracy First** - Strictly follows A-share trading rules (T+1, price limits, trading units)
- **Configuration Driven** - Control backtesting process via YAML config files
- **Easy to Extend** - Clean modular design for custom strategies
- **Complete Reports** - Auto-generate CSV, JSON, and chart reports
- **Command Line Tool** - Run complete backtest with a single command

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run backtest with default config
python backtest.py

# Run with custom config
python backtest.py --config my_config.yaml
```

## Project Structure

```
stocka/
├── quant_framework/    # Core framework code
├── examples/           # Example code
├── docs/               # Documentation
├── data/               # Data directory
├── reports/            # Report output
├── backtest.py         # CLI backtesting tool
├── config.yaml         # Configuration file
└── requirements.txt    # Dependencies
```

## Documentation

- [Configuration Guide](docs/CONFIG_GUIDE.md) - Detailed configuration instructions
- [Strategy Development Guide](docs/STRATEGY_GUIDE.md) - How to develop custom strategies
- [API Documentation](docs/API.md) - Core module API reference

## License

MIT License
