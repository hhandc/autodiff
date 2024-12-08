# Writing tests

Testing will be done using the `pytest` library.

The library first scans for `test_*.py` files. And within those files, it will scan for `test_*` function declarations.

Test functions should be using `assert` statements to check test values against expected values.

The main syntax is:

```
def test_something():
    test_val = 3 * 5
    true_val = 15
    assert test_val == true_val
```

But when you're comparing floats, this may not work to to rounding errors. In that case, you have to use `pytest.approx`:

```
from pytest import approx
def test_floats():
    my_float = 1.99999999 + 0.00000001
    true = 2.0
    assert my_float == approx(true)
```

## integration vs unit

Integration can be seen as tests that bunch together multiple features of the package. For example, if you're testing the autodiff in an actual usage setting, you will want to write an integration test.

On the other hand, unit tests are for testing a single function or a class. If you want to check that `Add` is correctly implemented, you would write an unit test for `Add`.