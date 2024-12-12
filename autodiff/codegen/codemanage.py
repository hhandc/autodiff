import os
import inspect

class manage_Code:
    def __init__(self, storage_dir="code_storage"):
        # Ensure a directory exists to store code files
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save(self, file_name: str, f: callable) -> str:
        code = inspect.getsource(f)
        
        file_path = os.path.join(self.storage_dir, file_name + ".py")
        with open(file_path, "w") as file:
            file.write(code)
        return file_path

    def retrieve(self, f: callable) -> str:
        file_path = os.path.join(self.storage_dir, f.__name__ + ".py")
        if not os.path.exists(file_path):
            return -1
        
        with open(file_path, "r") as file:
            code = file.read()
        return code