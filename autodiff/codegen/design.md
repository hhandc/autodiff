codegen works on the computation graph

1. Transform into single static assignment
    
    Identify reassignments

    ```
    a = x
    b = a
    a = y
    a + b  # this is y + x
    ```
    gets transformed into:
    ```
    a = x
    b = a
    a1 = y
    a1 + b
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