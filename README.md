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

## CLI Usage

Stocka provides a powerful command-line interface for managing data, factors, models, and live trading.

### Basic Usage

```bash
# Show help
python -m quant_framework.cli --help

# Show verbose output
python -m quant_framework.cli <command> -v
```

### Data Management

```bash
# Update stock data
python -m quant_framework.cli data update

# Update specific stocks
python -m quant_framework.cli data update --symbols 000001.SZ,600000.SH

# Update with date range
python -m quant_framework.cli data update --start 2024-01-01 --end 2024-12-31

# Check data status
python -m quant_framework.cli data status

# Show detailed data status
python -m quant_framework.cli data status --detail

# Initialize database
python -m quant_framework.cli data init

# Force reinitialize (warning: deletes existing data)
python -m quant_framework.cli data init --force
```

### Factor Calculation

```bash
# Calculate factors (default: alpha158)
python -m quant_framework.cli factor calculate

# Calculate specific factors for stocks
python -m quant_framework.cli factor calculate --symbols 000001.SZ --factor-name alpha158

# Calculate with date range
python -m quant_framework.cli factor calculate --start 2024-01-01 --end 2024-12-31

# Analyze factor performance
python -m quant_framework.cli factor analyze --factor-name alpha158

# Save analysis results
python -m quant_framework.cli factor analyze --factor-name alpha158 --output results.json

# List available factors
python -m quant_framework.cli factor list
```

### Machine Learning Pipeline

```bash
# Train model with default config
python -m quant_framework.cli ml train

# Train with custom config
python -m quant_framework.cli ml train --config my_config.yaml

# Train specific model
python -m quant_framework.cli ml train --model-name lightgbm --output models/my_model.pkl

# Backtest model
python -m quant_framework.cli ml backtest --model-path models/my_model.pkl

# Backtest with date range
python -m quant_framework.cli ml backtest --model-path models/my_model.pkl --start 2024-01-01 --end 2024-12-31

# Evaluate model
python -m quant_framework.cli ml evaluate --model-path models/my_model.pkl

# Evaluate specific metrics
python -m quant_framework.cli ml evaluate --model-path models/my_model.pkl --metrics sharpe_ratio,max_drawdown
```

### Live Trading

```bash
# Update real-time data (default 60 second interval)
python -m quant_framework.cli live update

# Update specific stocks
python -m quant_framework.cli live update --symbols 000001.SZ,600000.SH

# Set custom update interval (30 seconds)
python -m quant_framework.cli live update --interval 30

# Run as daemon
python -m quant_framework.cli live update --daemon

# Start live trading (dry-run mode)
python -m quant_framework.cli live trade --config config.yaml --dry-run

# Start real trading
python -m quant_framework.cli live trade --config config.yaml

# Check live trading status
python -m quant_framework.cli live status
```

### Command Structure

```
python -m quant_framework.cli <command> <subcommand> [options]

Commands:
  data      Data management operations
  factor    Factor calculation and analysis
  ml        Machine learning pipeline operations
  live      Real-time trading and data updates

Global Options:
  -v, --verbose    Show detailed log output
  --version        Show version information
  -h, --help       Show help message
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
