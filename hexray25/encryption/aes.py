import os
from pathlib import Path
from typing import Optional, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from pydantic import BaseModel

path_like = Union[str, Path]


class AESKey(BaseModel):
    key: Optional[bytes] = None
    iv: Optional[bytes] = None
    key_size: Optional[int] = 256

    def generate_key(self):
        self.key = os.urandom(
            self.key_size // 8
        )  # Divide by 8 to convert bits to bytes
        return self

    def generate_iv(self):
        self.iv = os.urandom(16)
        return self

    def generate_key_iv(self):
        self.key = os.urandom(
            self.key_size // 8
        )  # Divide by 8 to convert bits to bytes
        self.iv = os.urandom(16)
        return self

    def save_key_to_disk(self, save_path: path_like = Path("data") / "aes_key.bin"):
        with open(str(save_path), "wb") as f:
            f.write(self.key)
        return self

    def load_key_from_disk(self, load_path: path_like = Path("data") / "aes_key.bin"):
        with open(str(load_path), "rb") as f:
            self.key = f.read()
        return self

    def encrypt_data(self, *, data):
        # Create cipher configuration
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CFB(self.iv),
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()

        # Encrypt the data
        encrypted_data = encryptor.update(data) + encryptor.finalize()

        return self.iv, encrypted_data

    def decrypt_data(self, *, encrypted_data):
        # Create cipher configuration
        cipher = Cipher(
            algorithms.AES(self.key), modes.CFB(self.iv), backend=default_backend()
        )
        decryptor = cipher.decryptor()

        # Decrypt the data
        decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

        return decrypted_data

    def read_model_from_disk(self, load_path: path_like):
        with open(load_path, "rb") as f:
            model_data = f.read()
        return model_data

    def write_model_to_disk(self, *, save_path: path_like, data):
        with open(save_path, "wb") as f:
            f.write(self.iv + data)

    def read_encrypted_model_from_disk(self, load_path: path_like):
        with open(load_path, "rb") as f:
            self.iv = f.read(16)  # IV is the first 16 bytes
            encrypted_model = f.read()
        return encrypted_model

    def decrypt_model(self, *, encrypted_model):
        decrypted_model = self.decrypt_data(encrypted_data=encrypted_model)
        return decrypted_model
