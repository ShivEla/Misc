import socket
import os
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Random import get_random_bytes

def encrypt(data_bytes, key, mode):
    if mode == "ECB":
        cipher = AES.new(key, AES.MODE_ECB)
        padded_bytes = pad(data_bytes, AES.block_size)
        return cipher.encrypt(padded_bytes)

    elif mode == "CBC":
        cipher = AES.new(key, AES.MODE_CBC)
        padded_bytes = pad(data_bytes, AES.block_size)
        return cipher.iv + cipher.encrypt(padded_bytes)

    elif mode == "CFB":
        cipher = AES.new(key, AES.MODE_CFB)
        return cipher.iv + cipher.encrypt(data_bytes)
    
    else:
        raise ValueError("Invalid mode selected. Choose from ECB, CBC, CFB.")
def start_client():
    host, port = "127.0.0.1", 9999
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    print(f"\n--- Unified AES Client (Text & File) ---")
    
    try:
        client.connect((host, port))
        print(f"-> Connected to server at {host}:{port}")

        # --- Step 1: Receive the AES key from the server ---
        key = client.recv(16) 
        if not key:
            print("[!] Failed to receive AES key from server.")
            return
        print(f"-> Received AES Key (hex): {key.hex()}")

        # --- Step 2: Get user input for data type, content, and mode ---
        data_type_choice = input("Choose input type ('text' or 'file'): ").lower()
        mode = input("Choose AES Mode (ECB, CBC, CFB): ").upper()        
        original_data_bytes = None
        
        if data_type_choice == 'text':
            message = input("Enter message: ")
            original_data_bytes = message.encode('utf-8')
            print(f"\nOriginal Plaintext:   {message}")
        
        elif data_type_choice == 'file':
            filepath = input("Enter the full path to the file: ")
            if not os.path.exists(filepath):
                print(f"[!] File not found: {filepath}")
                return
            with open(filepath, "rb") as f:
                original_data_bytes = f.read()
            print(f"\nRead {len(original_data_bytes)} bytes from {os.path.basename(filepath)}")
            
        else:
            print("[!] Invalid input type. Exiting.")
            return

        # --- Step 3: Encrypt the data ---
        encrypted_data = encrypt(original_data_bytes, key, mode)
        print(f"Mode Selected:          {mode}")
        print(f"Ciphertext (hex):       {encrypted_data.hex()[:80]}...")

        # --- Step 4: Send metadata and encrypted data to the server ---
        header = f"{data_type_choice:<10}{mode:<10}".encode('utf-8')
        client.send(header)
        
        # Send the actual encrypted data
        client.send(encrypted_data)
        print("\n-> Encrypted data and metadata sent successfully.")
    except ConnectionRefusedError:
        print("[!] Connection refused. Is the server running?")
    except Exception as e:
        print(f"[!] An error occurred: {e}")
    finally:
        print("-> Closing client socket.")
        client.close()
if __name__ == "__main__":
    start_client()
