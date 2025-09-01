from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util import Counter
import binascii

# Helper to format bytes like the example (hex with spaces)
def to_hex(data):
    return " ".join(f"{b:02x}" for b in data)

# Function to test a mode
def test_mode(mode_name, cipher_factory, block_size=16):
    # Repeating pattern input (32 bytes)
    plaintext = bytes([(i % 16) for i in range(32)])
    
    # Create cipher
    cipher = cipher_factory()
    ciphertext = cipher.encrypt(plaintext)

    print(f"\nopmode: {mode_name}")
    print(f"input : {to_hex(plaintext)}")
    print(f"cipher: {to_hex(ciphertext)}")

    # ---- Test error propagation ----
    tampered = bytearray(ciphertext)
    tampered[5] ^= 0xFF  # Flip one byte
    cipher_dec = cipher_factory()  # Recreate cipher (new decryptor)
    decrypted = cipher_dec.decrypt(bytes(tampered))

    print(f"decrypted (tampered): {to_hex(decrypted)}")


# ----------------- MODES -----------------

key = get_random_bytes(16)
iv = get_random_bytes(16)

# ECB
test_mode("ECB", lambda: AES.new(key, AES.MODE_ECB))

# CBC
test_mode("CBC", lambda: AES.new(key, AES.MODE_CBC, iv=iv))

# CFB
test_mode("CFB", lambda: AES.new(key, AES.MODE_CFB, iv=iv))

# OFB
test_mode("OFB", lambda: AES.new(key, AES.MODE_OFB, iv=iv))

# CTR
ctr = Counter.new(128)
test_mode("CTR", lambda: AES.new(key, AES.MODE_CTR, counter=ctr))
