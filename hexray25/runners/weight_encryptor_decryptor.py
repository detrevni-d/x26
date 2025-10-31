from pydantic import BaseModel
from typing import Optional
from hexray25.encryption.encryptor_decryptor import EncryptorDecryptor
from loguru import logger
from hexray25.configs.weight_encryptor import WeightEncryptorConfig
import torch
import io

class _WeightEncryptor(BaseModel):
    model_path_raw: Optional[str] = None
    model_path_encrypted: Optional[str] = None

    def encrypt_model(self, ed: EncryptorDecryptor):
        ed.encrypt_file(file_path_raw=self.model_path_raw, file_path_encrypted=self.model_path_encrypted)
        logger.debug(f"Encrypted model from {self.model_path_raw} to {self.model_path_encrypted}")
    
    def decrypt_model(self, ed: EncryptorDecryptor):
        decrypted_data = ed.decrypt_file(load_path=self.model_path_encrypted)
        logger.debug(f"Decrypted model from {self.model_path_encrypted} to {self.model_path_raw}")
        state_dict = torch.load(io.BytesIO(decrypted_data), map_location="cpu")
        try:
            assert "hyper_parameters" in state_dict.keys(), "hyper parameters not present in state_dict"
            assert "state_dict" in state_dict.keys(), "state_dict not present in state_dict"
        except AssertionError as e:
            logger.error(f"Error in state_dict: {e}")
            return None
        return state_dict

class WeightEncryptor(BaseModel):
    params: WeightEncryptorConfig
    
    def encrypt_models(self):
        ed = EncryptorDecryptor(**self.params.ed_config.params)

        for model in self.params.models_config:
            name = model.name
            params = model.params
            model = _WeightEncryptor(model_path_raw=params["model_path_raw"], model_path_encrypted=params["model_path_encrypted"])
            model.encrypt_model(ed)
            logger.info(f"Encrypted model {name} is saved to {params['model_path_encrypted']}")
            

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

## secure communication 
many (clients) to one (server)

## read the code and figure out. 
1. where is signature used for encryption/decryption and why?
2. where is signature used for verification and why?

## list of runners 
1. setup encryption assets 
    - aes key
    - rsa key pair 
    - signature 
    - aes key encrypted by rsa private key 
    - signature encrypted.
2. encrypt model : use weightencryptor available here. 
    - bg model encrypted 
    - all defects model encrypted 
    - fmmd model encrypted 
3. decrypt file: - this goes into the ed and it gets called from the app. 
    - use encryptor_decryptor available in encryption folder. 
    - should take a .enc file and decrypt it to a dict 
4. verify signature 
"""