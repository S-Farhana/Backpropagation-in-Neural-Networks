import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd

hidden_size = 20  # Increased hidden neurons
learning_rate = 0.001  # Adjusted learning rate
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)

        # Initialize the biases
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, X):
        # Input to hidden layer
        self.hidden_activation = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_activation)

        # Hidden to output layer
        self.output_activation = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_activation)

        return self.predicted_output

    def backward(self, X, y, learning_rate):
        # Compute the output layer error
        output_error = y - self.predicted_output
        output_delta = output_error * self.sigmoid_derivative(self.predicted_output)

        # Compute the hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        loss_list = []  # To store the loss values for plotting
        for epoch in range(epochs):
            output = self.feedforward(X)
            self.backward(X, y, learning_rate)
            loss = np.mean(np.square(y - output))
            loss_list.append(loss)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

        return loss_list

    def test(self, X):
        return self.feedforward(X)

# MinMaxScaler equivalent function
def min_max_scaler(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val), min_val, max_val

# Manual inverse transform for the scaled data
def inverse_min_max_scaler(data, min_val, max_val):
    # Ensure the input data is a NumPy array
    data = np.array(data)
    return data * (max_val - min_val) + min_val

def prepare_data(stock_data, look_back):
    X, y = [], []
    for i in range(len(stock_data) - look_back):
        X.append(stock_data[i:i + look_back, 0])
        y.append(stock_data[i + look_back, 0])
    return np.array(X), np.array(y)


def load_stock_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['Date', 'Close']]  # Select the 'Close' column (closing stock prices)
    df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' to datetime
    df.set_index('Date', inplace=True)
    stock_data, min_val, max_val = min_max_scaler(df[['Close']].values)
    return stock_data, min_val, max_val

def prepare_data_for_regression(stock_data, look_back):
    X, y = [], []
    for i in range(len(stock_data) - look_back):
        X.append(stock_data[i:i + look_back].flatten())  # Flatten to include all previous days as features
        y.append(stock_data[i + look_back, 0])
    return np.array(X), np.array(y)

# Manual implementation of Multiple Linear Regression
class MultipleLinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        # Add a bias (intercept) column to the input features
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add a column of ones to X
        self.coefficients = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y  # Normal equation

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
        return X_b @ self.coefficients  # Make predictions


# Manual train-test split function
def train_test_split_manual(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test


def plot_loss(loss_list):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, color='blue', label='Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# Visualization function
def plot_predictions(actual_prices, nn_predicted_prices, mlr_predicted_prices, min_val, max_val):
    # Inversely transform the scaled data back to the original scale
    actual_prices = inverse_min_max_scaler(actual_prices, min_val, max_val).flatten()  # Flatten to 1D
    nn_predicted_prices = inverse_min_max_scaler(nn_predicted_prices, min_val, max_val).flatten()  # Flatten to 1D
    mlr_predicted_prices = inverse_min_max_scaler(mlr_predicted_prices, min_val, max_val).flatten()  # Flatten to 1D

    min_len = min(len(actual_prices), len(nn_predicted_prices), len(mlr_predicted_prices))
    actual_prices = actual_prices[:min_len]
    nn_predicted_prices = nn_predicted_prices[:min_len]
    mlr_predicted_prices = mlr_predicted_prices[:min_len]

    plt.figure(figsize=(12, 7))
    plt.style.use('seaborn-darkgrid')  

    # Plot actual stock prices
    plt.plot(actual_prices, label='Actual Prices', color='green', marker='o', linestyle='--', markersize=6, linewidth=2)

    # Plot predicted stock prices from Neural Network
    plt.plot(nn_predicted_prices, label='NN Predicted Prices', color='red', marker='x', linestyle='-', markersize=6, linewidth=2)

    # Plot predicted stock prices from Multiple Linear Regression
    plt.plot(mlr_predicted_prices, label='MLR Predicted Prices', color='blue', marker='s', linestyle='-', markersize=6, linewidth=2)

    
    plt.title("Stock Price Prediction: Actual vs Predicted", fontsize=18, fontweight='bold', color='blue')
    plt.xlabel("Time (Days)", fontsize=14, color='darkblue')
    plt.ylabel("Stock Price (Scaled)", fontsize=14, color='darkblue')

    plt.grid(True, which='both', linestyle='--', linewidth=0.7)

    plt.xticks(np.arange(0, min_len, step=5), fontsize=12, color='darkblue')
    plt.yticks(fontsize=12, color='darkblue')

    plt.legend(loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.show()


#Expression evaluation 
def evaluate(x_val, y_val, expression):
    x, y = sp.symbols('x y')
    expr = sp.sympify(expression)
    return float(expr.subs({x: x_val, y: y_val}))

#Runge Kutta Method
def runge_kutta_4(f, x0, y0, h):
    """4th-order Runge-Kutta method for a single step."""
    k1 = h * evaluate(x0, y0, f)
    k2 = h * evaluate(x0 + h / 2, y0 + k1 / 2, f)
    k3 = h * evaluate(x0 + h / 2, y0 + k2 / 2, f)
    k4 = h * evaluate(x0 + h, y0 + k3, f)
    
    y_next = y0 + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_next

#Adam's predictor
def adams_predictor(f_values, y_val, h):
    return y_val[3] + (h / 24) * (55 * f_values[3] - 59 * f_values[2] + 37 * f_values[1] - 9 * f_values[0])

#Adam's corrector
def adams_corrector(x_new, expression, f_values, predicted_y, y_val, h, tol=1e-6, max_iterations=10):
    for iteration in range(max_iterations):
        f_new = evaluate(x_new, predicted_y, expression)
        corrected_y = y_val[3] + (h / 24) * (9 * f_new + 19 * f_values[3] - 5 * f_values[2] + f_values[1])
        if abs(corrected_y - predicted_y) < tol:
            return corrected_y
        predicted_y = corrected_y

    return corrected_y

#Milne's Predictor
def milne_predictor(f_values, y_val, h):
    return y_val[0] + (4 * h / 3) * (2 * f_values[3] - f_values[2] + 2 * f_values[1])

#Milne's Corrector
def milne_corrector(x_new, expression, f_values, predicted_y, y_val, h, tol=1e-6, max_iterations=10):
    for iteration in range(max_iterations):
        f_new = evaluate(x_new, predicted_y, expression)
        corrected_y = y_val[2] + (h / 3) * (f_new + 4 * f_values[3] + f_values[2])
        if abs(corrected_y - predicted_y) < tol:
            return corrected_y
        predicted_y = corrected_y

    return corrected_y

#Jacobian 
def compute_jacobian(f1, f2, x, y):
    J = sp.Matrix([[sp.diff(f1, x), sp.diff(f1, y)], [sp.diff(f2, x), sp.diff(f2, y)]])
    return sp.lambdify((x, y), J, 'numpy')

# Rosenbrock-Euler method for solving system of ODEs
def rosenbrock_euler(f1_func, f2_func, J_func, x0, y0, t0, tf, h):
    t_values = np.arange(t0, tf + h, h)
    x_values = np.zeros_like(t_values)
    y_values = np.zeros_like(t_values)
    x_values[0] = x0
    y_values[0] = y0

    for i in range(1, len(t_values)):
        tn = t_values[i - 1]
        xn = x_values[i - 1]
        yn = y_values[i - 1]
        J = J_func(xn, yn)
        k1 = f1_func(xn, yn, tn) / (1 - h * J[0, 0])
        k2 = f2_func(xn, yn, tn) / (1 - h * J[1, 1])
        x_values[i] = xn + h * k1
        y_values[i] = yn + h * k2

    return t_values, x_values, y_values

# Function to compute Chebyshev nodes
def chebyshev_nodes(N):
    return np.cos(np.pi * np.arange(N + 1) / N)

# Function to compute Chebyshev differentiation matrix
def chebyshev_diff_matrix(N):
    if N == 0:
        return np.zeros((1, 1)), np.array([1])
    
    x = chebyshev_nodes(N)
    c = np.hstack([2, np.ones(N-1), 2]) * (-1)**np.arange(N+1)
    X = np.tile(x, (N+1, 1))
    dX = X - X.T
    
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N+1))
    D = D - np.diag(np.sum(D, axis=1))
    
    return D, x

# Gauss-Seidel iterative solver for the system
def gauss_seidel(D2, rhs, u, max_iter=1000, tol=1e-10):
    N = len(rhs)
    for iteration in range(max_iter):
        u_new = u.copy()
        for i in range(1, N-1):  # Interior points only
            u_new[i] = (rhs[i] - D2[i, :i] @ u_new[:i] - D2[i, i+1:] @ u[i+1:]) / D2[i, i]
        if np.linalg.norm(u_new - u, ord=np.inf) < tol:
            print(f"Converged in {iteration} iterations")
            break
        u = u_new
    return u

def __main__():
    while(1):
        print("----------------------------------------------------------")
        print("                            MENU                          ")
        print("----------------------------------------------------------")
        print("\n1.Adam's Method \n2.Milne's Method\n3.Rosenbrock Method\n4.Spectral Method\n5.Stock analysis using neural networks (NN)\n")
        choice= input("Choose the method \t: ").lower()
        if(choice=='adam' or choice=='milne'):
            expression = input("Enter the differential equation in terms of x and y (e.g., 'x + y**2'): ")
            x0 = float(input("Enter the initial x value: "))
            y0 = float(input("Enter the initial y value: "))
            h = float(input("Enter the step size h: "))
            x_values = [x0 + i * h for i in range(4)]
            y_val = [y0]  
            for i in range(1, 4):
                y_next = runge_kutta_4(expression, x_values[i-1], y_val[-1], h)
                y_val.append(y_next)
            f_values = [evaluate(x, y, expression) for x, y in zip(x_values, y_val)]
            if choice=='adam':
                predicted_value = adams_predictor(f_values, y_val, h)
                print(f"Predicted Value (Adams)\t:{predicted_value:.5f}")
                corrected_value = adams_corrector(x_values[-1] + h, expression, f_values, predicted_value, y_val, h)
                print(f"Corrected Value (Adams)\t:{ corrected_value:.5f}")
                print("\n")
                # Graph
                plt.plot(x_values, y_val, label='Runge-Kutta Initialization', color='blue', marker='o')
                plt.scatter(x_values[-1] + h, predicted_value, color='orange', label='Adams Predicted', marker='x')
                plt.scatter(x_values[-1] + h, corrected_value, color='red', label='Adams Corrected', marker='s')
                
            elif choice=='milne':
                predicted_value = milne_predictor(f_values, y_val, h)
                print(f"Predicted Value (Milne)\t:{predicted_value:.5f}" )
                corrected_value = milne_corrector(x_values[-1] + h, expression, f_values, predicted_value, y_val, h)
                print(f"Corrected Value (Milne)\t:{ corrected_value:.5f}")
                #Graph
                plt.plot(x_values, y_val, label='Runge-Kutta Initialization', color='blue', marker='o')
                plt.scatter(x_values[-1] + h, predicted_value, color='green', label='Milne Predicted', marker='x')
                plt.scatter(x_values[-1] + h, corrected_value, color='purple', label='Milne Corrected', marker='s')
            plt.xlabel('x', fontsize=14)
            plt.ylabel('y', fontsize=14)
            plt.title(f'{choice.capitalize()} choice: Prediction and Correction of ODE', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()  # Adjust layout
            plt.show()
        elif choice=='rosenbrock':
            x, y, t = sp.symbols('x y t')
            ode1_input = input("Enter the first ODE (dx/dt = f1(x, y, t)): ")
            ode2_input = input("Enter the second ODE (dy/dt = f2(x, y, t)): ")
            f1_sym = sp.sympify(ode1_input)
            f2_sym = sp.sympify(ode2_input)
            f1_func = sp.lambdify((x, y, t), f1_sym, 'numpy')
            f2_func = sp.lambdify((x, y, t), f2_sym, 'numpy')
            J_func = compute_jacobian(f1_sym, f2_sym, x, y)
            x0 = float(input("Enter the initial value of x0: "))
            y0 = float(input("Enter the initial value of y0: "))
            t0 = float(input("Enter the initial time t0: "))
            tf = float(input("Enter the final time tf: "))
            h = float(input("Enter the time step size h: "))
            t_values, x_values, y_values = rosenbrock_euler(f1_func, f2_func, J_func, x0, y0, t0, tf, h)

            # Display
            # Print formatted arrays
            print("\nTime values:")
            print(np.array2string(t_values, formatter={'float_kind': lambda x: "%.5f" % x}))

            print("x values:")
            print(np.array2string(x_values, formatter={'float_kind': lambda x: "%.5f" % x}))

            print("y values:")
            print(np.array2string(y_values, formatter={'float_kind': lambda x: "%.5f" % x}))            
            # Plot the results
            plt.plot(t_values, x_values, label="x(t)")
            plt.plot(t_values, y_values, label="y(t)")
            plt.xlabel("t")
            plt.ylabel("Values")
            plt.legend()
            plt.title("Rosenbrock-Euler Method Solution for System of ODEs")
            plt.grid(True)
            plt.show()
        elif choice=='spectral':
            N = int(input("Enter the number of Chebyshev points (e.g., 16): "))
            rhs_input = input("Enter the right-hand side function of the Poisson equation (e.g., sin(pi * x)): ")
            x_sym = sp.symbols('x')
            rhs_function = sp.lambdify(x_sym, sp.sympify(rhs_input), 'numpy')
            D, x = chebyshev_diff_matrix(N)
            D2 = D @ D
            D2_interior = D2[1:-1, 1:-1]
            x_interior = x[1:-1]
            rhs_interior = rhs_function(x_interior)
            u_interior = np.zeros_like(x_interior)
            u_interior = gauss_seidel(D2_interior, rhs_interior, u_interior)
            u = np.zeros(N + 1)
            u[1:-1] = u_interior
            # Plot the solution
            plt.figure(figsize=(12, 6))
            plt.plot(x, u, color='blue', linestyle='-', marker='o', markersize=6, label='Spectral Solution (Gauss-Seidel)')
            plt.xlabel('x', fontsize=14)
            plt.ylabel('u(x)', fontsize=14)
            plt.title('Poisson Equation Solution with Spectral Method (Gauss-Seidel)', fontsize=16)
            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add x-axis
            plt.axvline(0, color='black', linewidth=0.8, linestyle='--')  # Add y-axis
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.show()
        elif choice=='nn':
            file_path = r"C:\Users\farha\Downloads\EW-MAX (1).csv"  # Replace with the path to your stock data file
            stock_data, min_val, max_val = load_stock_data(file_path)

            look_back = 10  # Use the past 5 days to predict the next day
            X, y = prepare_data(stock_data, look_back)
            X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
            y_test = y_test.reshape(-1, 1)  

            # Create the neural network
            input_size = look_back 
            hidden_size = 10 
            output_size = 1 
            nn = NeuralNetwork(input_size, hidden_size, output_size)

            epochs = 10000
            learning_rate = 0.01
            loss_list = nn.train(X_train, y_train.reshape(-1, 1), epochs, learning_rate)

            # Generate predictions from the neural network
            nn_predictions = []
            current_input = X_train[-1].reshape(1, -1)  

            for _ in range(len(X_test)):
                nn_prediction = nn.test(current_input)
                nn_predictions.append(nn_prediction[0][0]) 
                current_input = np.append(current_input[:, 1:], nn_prediction).reshape(1, -1)  

            # Multiple Linear Regression Model
            mlr_model = MultipleLinearRegression()
            mlr_model.fit(X_train, y_train)

            linear_predictions = mlr_model.predict(X_test)

            nn_predictions = inverse_min_max_scaler(nn_predictions, min_val, max_val).flatten()
            linear_predictions = inverse_min_max_scaler(linear_predictions, min_val, max_val).flatten()
            
            # Plotting results for both predictions
            plot_predictions(y_test, nn_predictions, linear_predictions, min_val, max_val)  # Now this should show all prices correctly

        further=input("\nWould you like to perform further operations\t:").lower()
        if further!='yes':
            break
        
__main__()
        