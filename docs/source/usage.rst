Usage
=====

Here's a quick example using the STEM method:

.. code-block:: python

   import numpy as np
   from derivkit.hybrid import HybridDerivativeCalculator

    # Define a simple function
    def my_function(x):
        return np.array([np.sin(x)])

    # Initialize the calculator
    calc = HybridDerivativeCalculator(
        function=my_function,
        central_value=np.pi / 4,
        derivative_order=1
    )

    # Compute the derivatives using both methods
    stem_result = calc.stem_method()
    stencil_result = calc.five_point_stencil_method()

    # Print the results
    print("STEM method derivative:", stem_result)
    print("Five-point stencil derivative:", stencil_result)

Note on Vector Outputs

If your function returns a 2D array (e.g., shape `[n_x, n_y]`, such as a 2D data vector of angular power spectra in cosmology),
make sure to **flatten it** before passing it into the calculator.

The internal logic assumes each function evaluation returns a **1D array**, so you'll want to wrap your function like this:


.. code-block:: python

    def wrapped_func(x):
        return original_func(x).flatten()
    # After computing the derivative, you can reshape it back for interpretability:
    flat_deriv = calc.stem_method()
    reshaped = flat_deriv.reshape(n_x, n_y)


Explore the full API and examples in the Notebooks section or on GitHub.
