# Underfitting in Machine Learning

## What is Underfitting?

Underfitting occurs when a machine learning model is **too simple** to capture the underlying patterns in the data. The model fails to learn from the training data adequately and performs poorly on both training and unseen data.

Think of underfitting like studying only the chapter titles for an exam - you'll miss most of the important details and perform poorly on both practice tests and the actual exam.

## Visual Understanding of Underfitting

![image](https://github.com/user-attachments/assets/d2f7bb3b-21f6-4919-b454-48dba42d1afc)


*Figure 1: Illustration of underfitting in regression*

In the above illustration:
- **Green line (linear model)**: Too simple to capture the non-linear relationship in the data
- **Blue dashed line**: The true underlying pattern (quadratic relationship)
- **Red lines**: Show the large errors between predictions and actual values

## Real-World Analogy

Imagine teaching a child to identify animals:
- **Underfitting**: "All animals are either cats or dogs"
- **Good fit**: "Animals can be mammals, birds, reptiles, fish, etc., each with distinct characteristics"
- **Overfitting**: Memorizing every specific animal in a picture book without understanding general categories

## Common Signs of Underfitting

1. **High bias**: Model makes strong assumptions about the data that are incorrect
2. **Poor training accuracy**: Model performs poorly even on training data
3. **Poor testing accuracy**: Model performs poorly on new, unseen data
4. **Oversimplification**: Model is too simple to capture important patterns

## Mathematical Perspective

Underfitting can be understood from a bias-variance trade-off perspective:

- **Bias**: Error from erroneous assumptions in the learning algorithm
- **Variance**: Error from sensitivity to small fluctuations in the training set

Underfitting → High bias, low variance

## Code Example: Identifying Underfitting

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

# Generate sample quadratic data
np.random.seed(0)
X = np.sort(np.random.rand(30) * 5)
y = 1.5 * X**2 - 2 * X + 1 + np.random.normal(0, 1.0, size=X.shape)

# Reshape for sklearn
X = X.reshape(-1, 1)

# Create models with different complexity
models = [
    {"name": "Linear (Underfitting)", "degree": 1, "color": "green"},
    {"name": "Quadratic (Good Fit)", "degree": 2, "color": "blue"},
    {"name": "Degree 10 (Overfitting)", "degree": 10, "color": "red"}
]

# Plot the data and models
plt.figure(figsize=(12, 8))
plt.scatter(X, y, s=50, color='black', label='Data Points')

# Testing data for smooth curve visualization
X_test = np.linspace(0, 5, 100).reshape(-1, 1)

for model_info in models:
    degree = model_info["degree"]
    
    # Create and fit the model
    model = make_pipeline(
        PolynomialFeatures(degree=degree),
        LinearRegression()
    )
    model.fit(X, y)
    
    # Predict on test data
    y_pred = model.predict(X_test)
    
    # Calculate training error
    y_train_pred = model.predict(X)
    train_mse = mean_squared_error(y, y_train_pred)
    
    # Plot the model
    plt.plot(X_test, y_pred, 
             color=model_info["color"], 
             linewidth=2, 
             label=f"{model_info['name']} (MSE: {train_mse:.2f})")

plt.title("Underfitting vs. Good Fit vs. Overfitting", fontsize=16)
plt.xlabel("Feature Value", fontsize=14)
plt.ylabel("Target Value", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('fitting_comparison.png')
plt.show()
```

This code demonstrates three different models:
1. **Linear model (degree 1)**: Underfits the quadratic data
2. **Quadratic model (degree 2)**: Properly fits the underlying pattern
3. **High-degree polynomial (degree 10)**: Overfits the data

## How to Detect Underfitting

1. **High Training Error**: Poor performance on training data is the most obvious sign
2. **Similar Error Metrics**: Training and validation errors are similarly high
3. **Learning Curves**: Error plateaus at a high value even with more training data
4. **Feature Analysis**: Important features have little influence on model predictions

## Techniques to Address Underfitting

### 1. Increase Model Complexity
- Use more complex algorithms
- Add more parameters
- Use non-linear models for non-linear problems

```python
# Example: Switching from linear to polynomial regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Linear model (will underfit if data is non-linear)
linear_model = LinearRegression()

# Polynomial model (reduces underfitting for non-linear data)
polynomial_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)
```

### 2. Feature Engineering
- Create new features from existing ones
- Add interaction terms between features
- Transform features to better represent relationships

```python
# Example: Adding interaction terms
X_with_interactions = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
```

### 3. Reduce Regularization
- Lower the regularization strength if it's constraining the model too much

```python
# Example: Reducing regularization in Ridge regression
from sklearn.linear_model import Ridge

# Strong regularization (may cause underfitting)
strong_reg = Ridge(alpha=10.0)

# Reduced regularization (helps with underfitting)
weak_reg = Ridge(alpha=0.1)
```

### 4. Add More Features
- Collect additional relevant features that might explain the target variable

```python
# Conceptual example of adding features
X_extended = np.hstack([X, new_feature])
```

### 5. Use Ensemble Methods
- Combine multiple simple models to create a more complex one

```python
# Example: Using Random Forest instead of Decision Tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Simple model (may underfit)
tree_model = DecisionTreeRegressor(max_depth=3)

# More complex ensemble (reduces underfitting)
forest_model = RandomForestRegressor(n_estimators=100)
```

### 6. Deep Learning for Complex Problems
- Neural networks can learn complex patterns given enough data

```python
# TensorFlow/Keras example
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Creating a deeper neural network to reduce underfitting
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_size,)),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer
])
```

## Learning Curves for Identifying Underfitting

Learning curves plot training and validation errors against training set size:

```python
from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    # Calculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="blue")
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color="orange")
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="orange", label="Cross-validation Score")
    plt.legend(loc="best")
    
    return plt

# Example usage
underfitting_model = LinearRegression()  # Simple model for non-linear data
plot_learning_curve(underfitting_model, X, y, "Learning Curve (Underfitting Model)")
```

In an underfitting scenario, the learning curve will show:
- High error (low score) for both training and validation
- Small gap between training and validation scores
- Both scores plateau at a poor value as data size increases

## Practical Comparison: Underfitting vs. Good Fit

| Characteristic | Underfitting Model | Good Model |
|----------------|-------------------|------------|
| Training Error | High | Low-Moderate |
| Testing Error | High | Low-Moderate |
| Error Gap | Small | Small |
| Complexity | Too Low | Appropriate |
| Model Flexibility | Insufficient | Balanced |
| Assumptions | Too Strong | Reasonable |

## Example Case Studies

### Image Classification
A simple linear classifier trying to distinguish between 10 different types of animals would likely underfit, as animal classification requires capturing complex patterns.

### Text Analysis
Using a bag-of-words approach without considering word order or context often underfits for sentiment analysis in complex sentences.

### Medical Diagnosis
Using only basic vital signs to predict complex medical conditions will likely underfit since the relationship between symptoms and diseases is typically complex and non-linear.

## Balancing Underfitting and Overfitting

Finding the sweet spot between underfitting and overfitting is critical in machine learning:

1. **Start Simple**: Begin with simple models and gradually increase complexity
2. **Validation**: Always use validation data to monitor performance
3. **Learning Curves**: Plot learning curves to identify underfitting or overfitting
4. **Cross-Validation**: Use k-fold cross-validation to get reliable estimates of model performance
5. **Regularization Tuning**: Adjust regularization parameters to find the right balance

## Conclusion

Underfitting represents the "too simple" problem in machine learning. While Occam's razor suggests simpler explanations are better, models that are too simple fail to capture important patterns in the data.

Remember:
- **Complexity matters**: Ensure your model is complex enough for the problem
- **Feature engineering**: Create informative features that help the model learn
- **Model selection**: Choose appropriate algorithms for your data
- **Balance is key**: Find the sweet spot between underfitting and overfitting

