import numpy as np
import matplotlib.pyplot as plt


def gradient_descent(f, df, x0, step_size_fn, max_iter):
    """
    Performs gradient descent on the given function f, with its gradient df.

    :param f: A function whose input is an x, a column vector, and returns a scalar.
    :param df: A function whose input is an x, a column vector, and returns a column vector representing the gradient of f at x.
    :param x0: An initial value of x, x0, which is a column vector.
    :param step_size: The step size to use in each step
    :param max_iter: The number of iterations to perform

    :return x: the value at the final step
    :return fs: the list of values of f found during all the iterations (including f(x0))
    :return xs: the list of values of x found during all the iterations (including x0)
    """

    # Exercise 1 (d): Todo: Implement here.
    prev_x = x0
    fs = []
    xs =[]

    for i in range(max_iter):
        prev_f = f(prev_x)
        prev_grad = df(prev_x)
        fs.append(prev_f)
        xs.append(prev_x)
        
        if i == max_iter-1:
            return prev_x,fs,xs
        step = step_size_fn(i)
        prev_x = prev_x - step * prev_grad
    return prev_x, fs, xs


def transform_polynomial_basis_1d(x, order):
    """
    Transforms a single 1-dimensional data point x with a polynomial basis transformation.

    :param x: A numpy array with a single value (d=1).
    :param order: Can be 0, 1, 2 or 3.
    :return: The transformed data point x as a list.
    """
    if order==0:
        return [1]
    if order==1:
        return [1, x]
    if order==2:
        # Todo: Implement the polynomial basis for k=2:
        return [1, x, x**2]
    if order==3:
        # Todo: And for k=3:
        return [1, x, x**2, x**3]


def data_linear_trivial():
    X = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])
    Y = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])
    return X, Y


def data_linear_simple():
    X = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])
    Y = np.array([[-5], [-5], [-3], [1], [2], [5], [6]])
    return X, Y


def data_linear_offset():
    X = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])
    Y = np.array([[-6], [-4], [-2], [0], [2], [4], [6]]) + 6
    return X, Y


def data_quadratic():
    X = np.array([[-6], [-4], [-2], [0], [2], [4], [6]])
    Y = np.array([[12], [3], [1], [0], [1], [3], [12]])
    return X, Y


def data_polynomial():
    X = np.array([[-6], [-4], [-2], [0], [2], [4], [6], [8]])
    Y = np.array([[12], [3], [1], [0], [1], [3], [12], [3]])
    return X, Y


def plot_line_2d(axes, theta, line_style, xmin=-10, xmax=10):
    """
    Takes a 2-dimensional theta and plots the line defined by (theta[1], theta[0]) into the given axes.
    """
    p1_y = theta[0] * xmin + theta[1]
    p2_y = theta[0] * xmax + theta[1]
    ax.plot([xmin, xmax], [p1_y.flatten(), p2_y.flatten()], line_style)


if __name__ == '__main__':
    """
    We'll implement gradient descent, and test it on a simple quadratic function. We'll go on implementing the
    closed-form OLS solution, as well as solving the OLS objective with gradient descent, with various provided data.
    We'll analyse the convergence of gradient descent with various step sizes, and plot the loss (convergence plot).
    The final exercise uses polynomial basis transformation to fit more complex data with linear hypotheses.
    """

    # Exercise 1: Gradient descent to find the minimum of a simple quadratic function
    # -----------
    def f1(x):
        # Our function is f(x) = (3x+4)^2
        return float((3 * x + 4) ** 2)

    def df1(x):
        # 1 (c): Todo: Implement the derivative here
        return float(18 * x + 24)

    # 1 (e): Todo: Plot the function:
    fig_simple, ax_simple = plt.subplots()  # create an empty plot and retrieve the 'ax' handle
    f1_plot_x = np.arange(-10,10,0.05)
    f1_plot_y = np.array([f1(x) for x in f1_plot_x])
    ax_simple.plot(f1_plot_x,f1_plot_y)
    # ...

    # Todo: Set and plot the initial value:
    x0 = 0
    # ...

    # Run gradient descent to find the minimum of the function:
    # Todo: Experiment with step size, max_iter
    last_x, fs, xs = gradient_descent(f1, df1, x0, step_size_fn=lambda i: 0.01, max_iter=100)

    # Todo: Plot the found 'x' value and f(x)
    # ...
    ax_simple.plot(last_x,f1(last_x),'gx')
    # Todo: Plot each step of gradient descent, to see how it converges/diverges
    # ...


    # Exercise 2: Least Squares Regression
    # -----------
    # Get some example data (browse the file to see the various data_* functions provided):
    X, Y = data_quadratic()

    # Create a plot and set up some default plot options:
    fig, ax = plt.subplots()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.grid(True, which='both')
    ax.axhline(color='black', linewidth=0.5)
    ax.axvline(color='black', linewidth=0.5)
    ax.set_title("Least squares regression")
    # Todo: Plot the data here (ex. 2.2)
    ax.plot(X, Y, '+')

    # Todo: Feature transformation, add column of ones
    #X_augmented = np.insert(X,X.shape[1],[1],axis=1)
    X_augmented = np.array([transform_polynomial_basis_1d(x[0],order=2) for x in X])
    # Exercise 2.2: Todo: Compute theta* using the analytical OLS solution:
    # ------------
    theta_star = np.linalg.inv(X_augmented.T @ X_augmented) @ X_augmented.T @ Y

    # Todo: Plot the resulting hypothesis into the plot:
    #plot_line_2d(ax, theta_star,'g-')

    # Exercise 2.3 - Solution using gradient descent:
    # ------------

    # Todo: Implement the loss function:
    def squared_loss(x, y, theta):
        return (np.dot(theta.T,x)-y)**2

    # Todo: Implement the OLS objective function (using the loss):
    def ols_objective(X, Y, theta):
        return np.mean((np.dot((np.dot(X,theta)-Y).T,np.dot(X,theta)-Y)))

    # Todo: Implement the partial derivative of the squared loss w.r.t. theta
    def d_squared_loss_theta(x, y, theta):
        return (2/ x.shape[0]) * x.T @ (x @theta -y)

    # Todo: Implement the partial derivative of the OLS objective w.r.t. theta (using the partial derivative of the squared loss):
    def d_ols_objective_theta(x, y, theta):
        return d_squared_loss_theta(x,y,theta)

    # Finally, the gradient of our OLS objective is simply d_ols_objective_theta (as theta is our only parameter):
    def ols_objective_grad(X, Y, theta):
        return d_ols_objective_theta(X, Y, theta)

    # And we define the function that we want to minimise as the OLS objective over our dataset (X_augmented, Y):
    def f_ols(theta):
        return ols_objective(X_augmented, Y, theta)

    # And its gradient:
    def df_ols(theta):
        return ols_objective_grad(X_augmented, Y, theta)

    # Todo: Set an initial value for theta_init:
    theta_init = np.zeros((X_augmented.shape[1],1))

    # We define a step size function - let's return a constant step size, independent of the iteration i:
    def step_size_fn(i):
        return 0.001  # Todo: Experiment with various step sizes
    # Now we're ready to run gradient descent to minimise f_ols:
    last_x, fs, xs = gradient_descent(f_ols, df_ols, theta_init, step_size_fn=step_size_fn, max_iter=250)

    # Todo: Plot the found hypothesis into the figure with the data.
    # Todo: Also plot individual steps of gradient descent, to see how the optimisation behaves.
    plot_line_2d(ax, last_x, 'r-')
    # ...

    # Exercise 2.3 iii):
    fig_loss, ax_loss = plt.subplots()  # Create an empty figure for the loss plot
    # Todo: Plot the loss over the iterations
    # ... ax_loss.plot(...) ...

    # Optional: Exercise 2.4
    # Ex. 2.4 (b) iii): Plot the polynomial separator in 2D:
    x_plot = np.arange(-10,10,0.1)
    y_plot = []
    for x_p in x_plot:
        x_p_aug = np.array(transform_polynomial_basis_1d(x_p, order=2))
        y_p = theta_star.T @ x_p_aug.T
        y_plot.append(y_p)
    
    ax.plot(x_plot, y_plot, 'g-')


    print("Finished.")