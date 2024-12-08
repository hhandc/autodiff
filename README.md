## Installing

### Prerequisites
- python3(>= 3.10)
- run `pip3 install -r requirements.txt`

### How to install

- Create a virtual environment:
```$ python3 -m venv venv/```
- Activate the environment and install the package in editable mode:
```
$ source venv/bin/activate
$ pip3 install -e .
```
- To run tests, run `pytest tests/`

Note that **everytime before you start developing, you have to enable the venv** by running `source venv/bin/activate` in your terminal.