from typing import Optional
from pydantic import BaseModel
from hexray25.encryption.aes import AESKey
from hexray25.encryption.rsa import RSAKeyPair
import fastcore.all as fc 
from loguru import logger
from hexray25.configs.weight_encryptor import WeightEncryptorConfig

class SetupEncryption(BaseModel):
    params: WeightEncryptorConfig

    def setup(self):
        """
        1. setup encryption assets 
        - aes key
        - rsa key pair 
        - signature 
        - aes key encrypted by rsa private key 
        - signature encrypted.
        """ 
        #breakpoint()
        folder = fc.Path(self.params.folder)
        fc.Path(folder).mkdir(parents=True, exist_ok=True)
        aes_key = AESKey()
        aes_key.generate_key()
        logger.info(f"AES key generated: {aes_key.key}")
        aes_key.save_key_to_disk(save_path=folder / self.params.aes_key_path_raw)
        logger.info(f"AES key saved to {folder / self.params.aes_key_path_raw}")
        
        rsa_keypair = RSAKeyPair()
        rsa_keypair.generate_rsa_key_pair()
        logger.info("RSA key pair generated")
        rsa_keypair.save_keys_to_disk(save_path=folder)
        logger.info(f"RSA key pair saved to {folder}")

        # Now encrypt the aes key with the rsa public key
        aes_key.key = rsa_keypair.encrypt_aes_key(aes_key=aes_key.key)
        aes_key.save_key_to_disk(save_path=folder / self.params.aes_key_path_encrypted)
        logger.info(f"AES key encrypted and saved to {folder / self.params.aes_key_path_encrypted}")
        # create signature
        data = open(self.params.signature_path_raw, "rb").read()
        signature_encrypted = rsa_keypair.sign_data(data=data)
        open(folder / self.params.signature_path_encrypted, "wb").write(signature_encrypted)
        logger.info(f"Signature encrypted and saved to {folder / self.params.signature_path_encrypted}")
        
        rsa_keypair.verify_signature(data=data, signature=signature_encrypted)
        logger.info("Signature verified")



