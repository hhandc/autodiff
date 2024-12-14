def beale_func(x = 3, y = 0.5):
    a = 1.5 - x + x * y
    b = a ** 2
    a = 2.25 - x + x * y ** 2
    c = a ** 2
    a = 2.625 - x + x * y ** 3
    d = a ** 2

    return b + c + d
