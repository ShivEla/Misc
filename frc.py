import socket
import os
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

def encrypt_data(data_bytes, public_key):
    try:
        cipher = PKCS1_OAEP.new(public_key)
        return cipher.encrypt(data_bytes)
    except ValueError as e:
        # This error often occurs if the message is too large for the key size.
        print(f"[!] Encryption Error: {e}")
        print("[!] The file or message might be too large for a single RSA block.")
        print("[!] For large files, a hybrid encryption (RSA+AES) is recommended.")
        return None

def start_client():
    host, port = '127.0.0.1', 9999
    print("RSA Client")

    # --- Step 1: Get User Input ---
    choice = input("Choose input type ('text' or 'file'): ").lower()
    original_data_bytes = None
    data_type = ''

    if choice == 'text':
        message = input("Enter your message: ")
        original_data_bytes = message.encode('utf-8')
        print(f"\n[+] Original Message: {message}")
        data_type = 'TEXT'

    elif choice == 'file':
        filepath = input("Enter the full path to the file: ")
        if not os.path.exists(filepath):
            print(f"[!] Error: File not found at '{filepath}'")
            return
        
        with open(filepath, "rb") as f:
            original_data_bytes = f.read()
        print(f"\n[+] Original File Content (first 100 bytes): {original_data_bytes[:100]}...")
        data_type = 'FILE'
        
    else:
        print("[!] Invalid choice. Please run again and select 'text' or 'file'.")
        return

    # --- Step 2: Connect and Perform RSA Exchange ---
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        print(f"\n Connecting to server at {host}:{port}...")
        client_socket.connect((host, port))
        print("   Connection successful.")

        # Receive the server's public key
        public_key_pem = client_socket.recv(4096)
        public_key = RSA.import_key(public_key_pem)
        print(" Received server's public key.")

        # Encrypt the data 
        encrypted_data = encrypt_data(original_data_bytes, public_key)
        
        if encrypted_data is None:
            return
        print(f"Data encrypted successfully ({len(encrypted_data)} bytes).")
        print(f"Encrypted : {encrypted_data.hex()[:80]}...")

        # --- Step 3: Send Data Type and Encrypted Content ---
        # 1. Send the data type first so the server knows what to expect.
        client_socket.send(data_type.encode('utf-8'))
        
        # 2. Wait for a brief moment to ensure the server processes the type.
        import time
        time.sleep(0.1)

        # 3. Send the actual encrypted data.
        client_socket.send(encrypted_data)
        print("Encrypted data sent to server.")

    except ConnectionRefusedError:
        print("\nConnection failed. Is the server running?")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        print("\n Closing client socket.")
        client_socket.close()

if __name__ == "__main__":
    start_client()
