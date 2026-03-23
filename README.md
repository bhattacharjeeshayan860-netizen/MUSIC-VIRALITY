# Music Virality System

A machine learning system for predicting and analyzing music virality on YouTube and other platforms.

## Project Structure

- **data/**: Raw and processed datasets
- **models/**: Trained models and artifacts
- **src/**: Source code
  - **api/**: YouTube and other API clients
  - **pipelines/**: Data collection and processing pipelines
  - **features/**: Feature engineering
  - **training/**: Model training and evaluation
  - **utils/**: Utility functions
- **dashboard/**: Streamlit dashboard for visualization
- **notebooks/**: Jupyter notebooks for exploration
- **config/**: Configuration files
- **tests/**: Unit tests

## Installation

```bash
make install
```

## Usage

Run the dashboard:
```bash
make run
```

Run tests:
```bash
make test
```

## Configuration

Create a `.env` file and add your API keys and configuration:
```
YOUTUBE_API_KEY=your_key_here
```

## License

MIT License
