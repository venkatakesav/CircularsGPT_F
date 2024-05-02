# CircularsGPT Repository
Our Experiments and Scripts for the CircularsGPT project in association with IIIT-H and IIT-B.

To achieve a neat and tidy directory structure for your CircularsGPT project, as well as write reusable and extensible object-oriented code, here's a suggested approach:

```
CircularsGPT/
├── data/
│   ├── raw/
│   │   └── pdfs/
│   └── processed/
│       ├── annotations/
│       └── synthetic/
├── models/
│   ├── layoutlm/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── utils.py
│   ├── udop/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── utils.py
│   ├── pix2struct/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── utils.py
│   └── __init__.py
├── metrics/
│   ├── __init__.py
│   └── metrics.py
├── utils/
│   ├── __init__.py
│   ├── data_utils.py
│   └── model_utils.py
├── main.py
├── requirements.txt
└── README.md
```

- `data/`: This directory will contain all the data files.
  - `raw/pdfs/`: This directory will store the raw PDF files scraped from the internet.
  - `processed/annotations/`: This directory will store the annotated data.
  - `processed/synthetic/`: This directory will store the synthetically generated data.
- `models/`: This directory will contain the different models you'll be using.
  - Each model will have its own subdirectory (e.g., `layoutlm/`, `udop/`, `pix2struct/`).
  - Inside each model subdirectory, there will be three files:
    - `__init__.py`: This file will be used to initialize the model package.
    - `model.py`: This file will contain the model class and its implementation.
    - `utils.py`: This file will contain utility functions specific to the model.
  - `__init__.py`: This file will be used to initialize the `models` package.
- `metrics/`: This directory will contain the evaluation metrics.
  - `__init__.py`: This file will be used to initialize the `metrics` package.
  - `metrics.py`: This file will contain the implementation of various evaluation metrics.
- `utils/`: This directory will contain utility functions shared across the project.
  - `__init__.py`: This file will be used to initialize the `utils` package.
  - `data_utils.py`: This file will contain utility functions related to data processing.
  - `model_utils.py`: This file will contain utility functions related to model training and evaluation.
- `main.py`: This file will be the entry point of your application, where you'll orchestrate the data processing, model training, and evaluation.
- `requirements.txt`: This file will list all the Python package dependencies for your project.
- `README.md`: This file will contain the documentation for your project.

To write reusable and extensible object-oriented code, you can follow these guidelines:

1. **Define a base model class**: Create a base model class that encapsulates the common functionality shared across all models. This class should define the basic methods for training, evaluation, and prediction. Example:

```python
# models/__init__.py
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def train(self, train_data):
        pass

    @abstractmethod
    def evaluate(self, eval_data):
        pass

    @abstractmethod
    def predict(self, input_data):
        pass
```

2. **Inherit from the base model class**: For each specific model (e.g., LayoutLM, UDOP, Pix2Struct), create a subclass that inherits from the `BaseModel` class and implements the abstract methods. Example:

```python
# models/layoutlm/model.py
from models import BaseModel

class LayoutLM(BaseModel):
    def __init__(self, config):
        self.config = config
        # Initialize the model

    def train(self, train_data):
        # Implement training logic

    def evaluate(self, eval_data):
        # Implement evaluation logic

    def predict(self, input_data):
        # Implement prediction logic
```

3. **Use dependency injection**: To ensure loose coupling and better testability, you can use dependency injection to inject the required dependencies (e.g., data loaders, model configurations) into your model classes. This will make it easier to swap out dependencies without modifying the model code.

4. **Write utility functions**: Whenever you identify a piece of reusable code, extract it into a utility function or module. This will promote code reuse and maintainability.

5. **Use configuration files**: Instead of hard-coding values like paths, hyperparameters, or other settings, use configuration files (e.g., JSON, YAML) to store these values. This will make it easier to change settings without modifying the code.

6. **Write tests**: As you develop your code, write unit tests to ensure the correctness of your implementations. This will make it easier to refactor or extend your code in the future without introducing regressions.

By following these guidelines, you'll end up with a well-structured, reusable, and extensible codebase that will make it easier to add new models, compare results, and maintain your CircularsGPT project in the long run.