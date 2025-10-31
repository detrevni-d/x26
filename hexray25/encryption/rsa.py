from pathlib import Path
from typing import Optional, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from pydantic import BaseModel

path_like = Union[str, Path]


class RSAKeyPair(BaseModel):
    public_key: Optional[str] = None  # PEM format, serialized
    private_key: Optional[str] = None  # PEM format, serialized
    key_size: Optional[int] = 2048
    public_exponent: Optional[int] = 65537

    def generate_rsa_key_pair(self):
        private_key = rsa.generate_private_key(
            public_exponent=self.public_exponent,
            key_size=self.key_size,
            backend=default_backend(),
        )
        public_key = private_key.public_key()

        # Serialize private key
        self.private_key = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        # Serialize public key
        self.public_key = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return self

    def save_keys_to_disk(self, save_path: path_like = Path("data")):
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
        with open(str(save_path / "rsa_key_private.pem"), "wb") as f:
            f.write(self.private_key)
        with open(str(save_path / "rsa_key_public.pem"), "wb") as f:
            f.write(self.public_key)
        return self

    def load_private_key_from_disk(
        self, path: path_like = Path("data/rsa_key_private.pem")
    ):
        if not isinstance(path, Path):
            path = Path(path)
        with open(str(path), "rb") as f:
            self.private_key = f.read()
        return self

    def load_public_key_from_disk(
        self, path: path_like = Path("data/rsa_key_public.pem")
    ):
        if not isinstance(path, Path):
            path = Path(path)
        with open(str(path), "rb") as f:
            self.public_key = f.read()
        return self

    def encrypt_aes_key(self, *, aes_key):
        public_key = serialization.load_pem_public_key(
            self.public_key, backend=default_backend()
        )

        encrypted_aes_key = public_key.encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return encrypted_aes_key

    def decrypt_aes_key(self, *, encrypted_aes_key):
        private_key = serialization.load_pem_private_key(
            self.private_key, password=None, backend=default_backend()
        )

        decrypted_aes_key = private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return decrypted_aes_key

    def sign_data(self, *, data):
        private_key = serialization.load_pem_private_key(
            self.private_key, password=None, backend=default_backend()
        )
        signature = private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256(),
        )
        return signature

    def verify_signature(self, *, data, signature):
        public_key = serialization.load_pem_public_key(
            self.public_key, backend=default_backend()
        )
        try:
            public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False
