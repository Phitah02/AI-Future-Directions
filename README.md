# AI Future Directions

**By Peter Mwaura**

This repository contains implementations and analyses for the AI Future Directions assignment, exploring cutting-edge AI technologies including Edge AI, Quantum AI, and AI-driven IoT systems.

## Project Overview

This project consists of two main tasks that demonstrate different aspects of modern AI applications:

1. **Task 1: Edge AI Prototype** - Garbage classification using TensorFlow Lite for edge deployment
2. **Task 2: AI-Driven IoT System** - Smart agriculture simulation with yield prediction using AI and IoT sensors

Additionally, the project includes comprehensive essay responses on Edge AI and Quantum AI topics.

## Project Structure

```
AI-Future-Directions/
├── Task 1/
│   ├── edge_ai_prototype.ipynb          # Main implementation notebook
│   ├── edge_ai_prototype.py             # Python script version
│   ├── garbage_classifier.tflite        # TensorFlow Lite model
│   ├── best_model.h5                    # Best trained Keras model
│   ├── Edge_AI_Report.md                # Detailed report
│   ├── Task 1.md                        # Task documentation
│   └── garbage_classification/          # Dataset directory
│       ├── battery/
│       ├── biological/
│       ├── brown-glass/
│       ├── cardboard/
│       ├── clothes/
│       └── ... (12 classes total)
│
├── Task 2/
│   ├── Smart_Agriculture_AI_IoT_Yield_Prediction.ipynb  # Main implementation
│   ├── Smart_Farming_Crop_Yield_2024.csv                # Dataset
│   ├── Task2_Smart_Agriculture_Design.md                # Design documentation
│   └── data flow diagram.mmd                            # System architecture diagram
│
├── essay questions.md                   # Essay responses on Edge AI and Quantum AI
├── README.md                            # This file
└── TODO.md                              # Project tasks and notes
```

## Task 1: Edge AI Prototype - Garbage Classification

### Overview
Implementation of a lightweight image classification model for recognizing recyclable items using TensorFlow Lite. This demonstrates Edge AI principles by training a MobileNetV2-based model and converting it to an efficient TFLite format for deployment on edge devices.

### Dataset
- **Source**: [Kaggle - Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- **Location**: `Task 1/garbage_classification`
- **Classes**: 12 categories (battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass)
- **Model**: MobileNetV2 (pre-trained on ImageNet) with custom classification head
- **Output**: TensorFlow Lite model optimized for edge deployment

### Key Features
- Real-time image classification on edge devices
- Low latency inference without cloud dependency
- Privacy-preserving (data stays on-device)
- Optimized model size for resource-constrained devices
- Support for offline operation

### Files
- `edge_ai_prototype.ipynb`: Complete implementation with training and conversion
- `garbage_classifier.tflite`: Deployable TensorFlow Lite model
- `best_model.h5`: Best trained Keras model checkpoint
- `Edge_AI_Report.md`: Comprehensive technical report

## Task 2: AI-Driven IoT System - Smart Agriculture Yield Prediction

### Overview
Design and implementation of a smart agriculture simulation system using AI and IoT to monitor and optimize crop growth. The system collects real-time data from sensors, processes it using AI models, and provides insights for predictive analytics such as yield prediction and irrigation optimization.

### Dataset
- **Source**: [Kaggle - Smart Farming Sensor Data for Yield Prediction](https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction)
- **Location**: `Task 2/Smart_Farming_Crop_Yield_2024.csv`
- **Features**: Sensor data including soil moisture, temperature, humidity, light intensity, pH, rainfall, wind, and CO2 levels
- **Model**: Machine Learning Regression Model (Random Forest Regressor / Gradient Boosting)
- **Output**: Crop yield predictions with confidence intervals

### Sensors Used
1. **Soil Moisture Sensor**: Measures water content in soil
2. **Temperature Sensor**: Monitors ambient and soil temperature
3. **Humidity Sensor**: Tracks relative humidity levels
4. **Light Intensity Sensor**: Measures sunlight exposure
5. **pH Sensor**: Monitors soil pH levels
6. **Rainfall Sensor**: Detects precipitation
7. **Wind Speed and Direction Sensor**: Assesses wind conditions
8. **CO2 Sensor**: Measures carbon dioxide levels

### Key Features
- Real-time sensor data collection and processing
- AI-powered yield prediction
- Decision support system for irrigation and resource management
- Data flow visualization and system architecture
- Predictive analytics for crop optimization

### Files
- `Smart_Agriculture_AI_IoT_Yield_Prediction.ipynb`: Complete implementation
- `Task2_Smart_Agriculture_Design.md`: System design documentation
- `data flow diagram.mmd`: Mermaid diagram of system architecture

## Essay Questions

The repository includes comprehensive essay responses covering:

1. **Edge AI vs Cloud-Based AI**: Analysis of how Edge AI reduces latency and enhances privacy, with real-world examples including autonomous drones.

2. **Quantum AI vs Classical AI**: Comparison of Quantum AI and classical AI in solving optimization problems, including industries that could benefit most from Quantum AI.

See `essay questions.md` for detailed responses.

## Technologies Used

### Task 1
- **TensorFlow/Keras**: Model training
- **TensorFlow Lite**: Edge deployment
- **MobileNetV2**: Pre-trained base model
- **Python**: Implementation language
- **Jupyter Notebook**: Development environment

### Task 2
- **Scikit-learn**: Machine learning models (Random Forest, Gradient Boosting)
- **Pandas/NumPy**: Data processing
- **Matplotlib/Seaborn**: Data visualization
- **Python**: Implementation language
- **Jupyter Notebook**: Development environment

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required Python packages (install via pip):
  ```bash
  pip install tensorflow pandas numpy matplotlib seaborn scikit-learn jupyter
  ```

### Running Task 1
1. Navigate to `Task 1/` directory
2. Open `edge_ai_prototype.ipynb` in Jupyter Notebook
3. Ensure the `garbage_classification/` dataset is in the same directory
4. Run all cells to train the model and generate the TFLite version

### Running Task 2
1. Navigate to `Task 2/` directory
2. Open `Smart_Agriculture_AI_IoT_Yield_Prediction.ipynb` in Jupyter Notebook
3. Ensure the dataset CSV file is in the same directory
4. Run all cells to train the model and generate predictions

## Key Concepts Demonstrated

### Edge AI Benefits
- **Low Latency**: Local processing eliminates network delays
- **Privacy**: Sensitive data stays on-device
- **Offline Operation**: Functions without internet connectivity
- **Reduced Bandwidth**: Minimizes data transmission
- **Cost Efficiency**: Reduces cloud computing costs
- **Energy Efficiency**: Optimized for resource-constrained devices

### AI-Driven IoT
- **Real-time Data Collection**: Continuous monitoring via sensors
- **Predictive Analytics**: AI models for forecasting and optimization
- **Decision Support**: Automated recommendations for resource management
- **Closed-loop Systems**: Feedback mechanisms for continuous improvement

## Results and Performance

### Task 1 (Edge AI)
- Model optimized for edge deployment
- Significant reduction in model size with TensorFlow Lite
- Fast inference times suitable for real-time applications
- High accuracy maintained despite optimization

### Task 2 (Smart Agriculture)
- Predictive models for crop yield forecasting
- Integration of multiple sensor data streams
- Decision support for agricultural optimization
- Scalable architecture for IoT deployment

## Future Improvements

### Task 1
- Advanced quantization techniques
- Hardware acceleration (Edge TPU, NPU)
- Continuous learning capabilities
- Multi-modal input integration

### Task 2
- Real-time sensor integration
- Advanced time-series models (LSTM, Transformer)
- Multi-crop support
- Integration with weather APIs
- Mobile application for farmers

## License

This project is part of an academic assignment. Please refer to the original dataset licenses:
- [Garbage Classification Dataset License](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)
- [Smart Farming Dataset License](https://www.kaggle.com/datasets/atharvasoundankar/smart-farming-sensor-data-for-yield-prediction)

## Author

**Peter Mwaura**

## Acknowledgments

- Kaggle for providing the datasets
- TensorFlow team for TensorFlow Lite
- Open-source community for various libraries and tools
