# PathFinder Navigation System

This project implements an AI-powered navigation system using reinforcement learning to navigate through 3D environments. The system is built on AI2-THOR framework for simulating indoor environments and provides a graphical user interface for interacting with the navigation agent.

## Features

- GUI-based environment exploration and navigation
- Neural network-based pathfinding using reinforcement learning
- Support for loading pretrained navigation models
- Visualization of navigation paths
- Manual and automatic navigation modes

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch
- PyQt6
- AI2-THOR
- OpenCV
- NumPy

### Setup

1. Clone the repository:
```
git clone https://github.com/yourusername/pathfinder.git
cd pathfinder
```

2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

### Running the Navigation System

1. Start the main application:
```
python application/main.py
```

2. Select a trained model file (`.pt` format)
3. Choose an environment from the dropdown
4. Click "Launch Navigation" to start the navigation interface

### Navigation Interface

The navigation interface consists of:
- A top-down map view showing the environment
- Controls for setting start and goal positions
- Buttons for manual and automatic navigation

#### Setting Positions

1. Click "Set Start" and then click on the map to select a starting position
2. Click "Set Goal" and then click on the map to select a goal position

#### Navigation Modes

- **Run Manual**: Navigate manually using keyboard controls (WASD)
- **Run Auto**: Let the AI navigate automatically to the goal

### Training a Navigation Model

To train a new navigation model:

```
python model/train_navigation.py
```

The trained model will be saved in the `trained_models` directory.

## Keyboard Controls

When in manual navigation mode:
- **W**: Move forward
- **S**: Move backward
- **A**: Face west
- **D**: Face east
- **Q**: Face north
- **E**: Face south

## Project Structure

- `application/`: Contains the GUI application code
  - `main.py`: Main application entry point
  - `gui/`: User interface components
  - `thor_controller/`: AI2-THOR environment controller
- `model/`: Contains model architecture and training code
  - `navigation_policy.py`: Neural network architecture
  - `navigation_trainer.py`: Training loop implementation
  - `train_navigation.py`: Script for training new models

## License

This project is licensed under the MIT License - see the LICENSE file for details.