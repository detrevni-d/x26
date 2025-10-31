from pydantic import BaseModel
from hexray25.encryption import AESKey, RSAKeyPair
from typing import Optional
from loguru import logger
import torch
import io

from transformers.models.convnextv2.configuration_convnextv2 import ConvNextV2Config
torch.serialization.add_safe_globals([ConvNextV2Config])

"""
two kind of encryption:
1. symmetric encryption: AES
2. asymmetric encryption: RSA


## this is the first time. 
1. create AES and RSA key pairs 
2. use AES to encrypt models weights
3. use RSA to encrypt AES key 
4. save RSA public key to disk 
5. save encrypted AES key to disk 

## everytime we have a new model 
1. use AES to encrypt models weights.


"""

class EncryptorDecryptor(BaseModel):
    aes_key_path_raw: Optional[str] = None
    aes_key_path_encrypted: Optional[str] = None
    rsa_key_path_private: Optional[str] = None
    rsa_key_path_public: Optional[str] = None
    signature_path_raw: Optional[str] = None
    signature_path_encrypted: Optional[str] = None 

    def encrypt_file(self, *, file_path_raw, file_path_encrypted):
        aes_key = AESKey()
        aes_key.load_key_from_disk(self.aes_key_path_raw)
        aes_key.generate_iv()

        model_data = aes_key.read_model_from_disk(file_path_raw)
        iv, encrypted_model = aes_key.encrypt_data(data=model_data)
        # Note: IV is stored with the encrypted data
        aes_key.write_model_to_disk(
            save_path=file_path_encrypted,
            data=encrypted_model,
        ) 
        logger.debug(f"Encrypted data is saved to {file_path_encrypted}")

    # def decrypt_file(self, *, file_path_encrypted, save_path_decrypted=None):
    #     with open(self.aes_key_path_encrypted, "rb") as f:
    #         encrypted_aes_key = f.read()
    #     rsa_key_pair = RSAKeyPair()
    #     rsa_key_pair.load_private_key_from_disk(self.rsa_key_path_private)
    #     decrypted_aes_key = rsa_key_pair.decrypt_aes_key(
    #         encrypted_aes_key=encrypted_aes_key
    #     )
    #     aes_key = AESKey(key=decrypted_aes_key)

    #     with open(file_path_encrypted, "rb") as f:
    #         encrypted_data = f.read()
    #     iv, decrypted_data = aes_key.decrypt_data(encrypted_data=encrypted_data)
        
    #     if save_path_decrypted is not None:
    #         with open(save_path_decrypted, "wb") as f:
    #             f.write(decrypted_data)
    #         logger.debug(f"Decrypted data is saved to {save_path_decrypted}")
    #     else:
    #         return decrypted_data
    
    def decrypt_file(self, *, load_path):
        rsa_key_pair = RSAKeyPair()
        rsa_key_pair.load_public_key_from_disk(self.rsa_key_path_public)
        rsa_key_pair.load_private_key_from_disk(self.rsa_key_path_private)
        #rsa_key_pair.private_key = private_key_pem

        # Construct AESKey object
        with open(self.aes_key_path_encrypted, "rb") as f:
            encrypted_aes_key = f.read()

        decrypted_aes_key = rsa_key_pair.decrypt_aes_key(encrypted_aes_key=encrypted_aes_key)
        aes_key = AESKey(key=decrypted_aes_key)

        # Decrypt defects weight file using the AESKey object
        encrypted_data = aes_key.read_encrypted_model_from_disk(load_path=load_path)
        decrypted_data = aes_key.decrypt_data(encrypted_data=encrypted_data)

        rsa_key_pair.private_key = None

        state_dict = torch.load(io.BytesIO(decrypted_data), map_location="cpu", weights_only=True)
        try:
            assert "hyper_parameters" in state_dict.keys(), "hyper parameters not present in state_dict"
            assert "state_dict" in state_dict.keys(), "state_dict not present in state_dict"
        except AssertionError as e:
            logger.error(f"Error in state_dict: {e}")
            return None
        return state_dict

    def encrypt_aes_key(self):
        #TODO: why are we generating rsa_key_pair here? 
        # aes_key = AESKey()
        # aes_key.load_key_from_disk(self.aes_key_path_raw)

        # rsa_keypair = RSAKeyPair()
        # rsa_keypair.generate_rsa_key_pair().save_keys_to_disk()

        # aes_key.key = rsa_keypair.encrypt_aes_key(aes_key=aes_key.key)
        # aes_key.save_key_to_disk(save_path=self.aes_key_path_encrypted)
        # logger.debug(f"Encrypted AES key: {aes_key.key}")
        pass 
    
    def decrypt_aes_key(self):
        aes_key = AESKey()
        aes_key.load_key_from_disk(self.aes_key_path_encrypted)

        rsa_keypair = RSAKeyPair()
        rsa_keypair.load_private_key_from_disk(self.rsa_key_path_private)
        aes_key.key = rsa_keypair.decrypt_aes_key(aes_key=aes_key.key)
        logger.debug(f"Decrypted AES key: {aes_key.key}")
        return aes_key

    def create_signature(self, *, signature: bytes = b"clear message"):
        rsa_keypair = RSAKeyPair()
        rsa_keypair.load_private_key_from_disk(self.rsa_key_path_private)
        rsa_keypair.load_public_key_from_disk(self.rsa_key_path_public)

        encrypted_signature = rsa_keypair.sign_data(data=signature)

        with open(self.signature_path_raw, "wb") as f:
            f.write(signature)
        with open(self.signature_path_encrypted, "wb") as f:
            f.write(encrypted_signature)

        logger.debug(f"Signature: {signature}")
        logger.debug(f"Encrypted signature: {encrypted_signature}")

        is_valid = rsa_keypair.verify_signature(
            data=signature, signature=encrypted_signature
        )
        if is_valid:
            logger.info("valid signature is created")
        else:
            logger.error("invalid signature")
