import os
import inspect

class manage_Code:
    def __init__(self, storage_dir="code_storage"):
        # Ensure a directory exists to store code files
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)

    def save(self, file_name: str, f: callable) -> str:
        """
        Saves the provided callable's source code into a file with the given name.
        Args:
            file_name (str): The name of the file to save the code.
            f (callable): The callable object to save (e.g., a function).
        Returns:
            str: The path to the saved code file.
        Raises:
            TypeError: If the provided object is not callable.
        """
        code = inspect.getsource(f)
        
        file_path = os.path.join(self.storage_dir, file_name + ".py")
        with open(file_path, "w") as file:
            file.write(code)
        return file_path

    def retrieve(self, file_name: str) -> str:
        """
        Retrieves the code from a file with the given name.
        Args:
            file_name (str): The name of the file to retrieve the code from.
        Returns:
            str: The retrieved code content.
        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = os.path.join(self.storage_dir, file_name + ".py")
        if not os.path.exists(file_path):
            return -1
        
        with open(file_path, "r") as file:
            code = file.read()
        return code