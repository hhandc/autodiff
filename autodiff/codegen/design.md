codegen works on the computation graph

1. Transform into single static assignment
    
    Identify reassignments

    ```
    a = x
    b = a
    a = y
    a + b  # this is x + y
    ```
    gets transformed into:
    ```
    a1 = x
    b = a1
    a2 = y
    a2 + y
    ```
2. Find what portions of forward evaluation are needed for backward calculation.
    
    For example, consider the following code:
    ```
    a = x * y
    b = a * y

    dx = 1.0
    dx *= y ** 2

    dy = 1.0
    dy *= 2y
    ```

    Backward pass needs `a, y` for evaluation. We can convert into:
    ```
    a = x * y
    b = a * y

    dx = 1.0
    ```