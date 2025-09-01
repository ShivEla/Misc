import socket
import os
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import unpad

def recv_all(sock, n):
    data = b""
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

def decrypt(encrypted_data, key, mode):
    try:
        if mode == "ECB":
            cipher = AES.new(key, AES.MODE_ECB)
            decrypted_padded = cipher.decrypt(encrypted_data)
            return unpad(decrypted_padded, AES.block_size)

        elif mode == "CBC":
            iv = encrypted_data[:AES.block_size]
            ciphertext = encrypted_data[AES.block_size:]
            cipher = AES.new(key, AES.MODE_CBC, iv=iv)
            decrypted_padded = cipher.decrypt(ciphertext)
            return unpad(decrypted_padded, AES.block_size)

        elif mode == "CFB":
            iv = encrypted_data[:AES.block_size]
            ciphertext = encrypted_data[AES.block_size:]
            cipher = AES.new(key, AES.MODE_CFB, iv=iv)
            return cipher.decrypt(ciphertext)
            
        else:
            raise ValueError("Invalid mode received.")
            
    except (ValueError, KeyError) as e:
        print(f"[!] Decryption failed. The data may be corrupt or the key is wrong. Error: {e}")
        return None

def start_server():
    host, port = "127.0.0.1", 9999
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"\nAES Server")
    print(f"-> Listening on {host}:{port}")

    client, addr = server.accept()
    print(f"-> Accepted connection from {addr}\n")

    try:
        # --- Step 1: Generate and send the AES key ---
        key = get_random_bytes(16) # 128-bit key
        client.send(key)
        print(f"Generated AES Key (hex): {key.hex()}")

        # --- Step 2: Receive metadata header ---
        header = recv_all(client, 20)
        if not header:
            print("[!] Failed to receive header from client.")
            return
            
        data_type = header[:10].decode('utf-8').strip()
        mode = header[10:].decode('utf-8').strip()
        print(f"Data Type Received:      {data_type}")
        print(f"Mode Selected:           {mode}")

        # --- Step 3: Receive the encrypted data ---
        encrypted_data = client.recv(4096) 
        print(f"Ciphertext (hex):        {encrypted_data.hex()[:80]}...")

        # --- Step 4: Decrypt the data ---
        decrypted_bytes = decrypt(encrypted_data, key, mode)

        # --- Step 5: Process and Verify decrypted data ---
        if decrypted_bytes:
            print("\n[+] Decryption successful!")
            if data_type == 'text':
                message = decrypted_bytes.decode('utf-8')
                print("\n--- Verification ---")
                print(f"Decrypted Message: {message}")

            elif data_type == 'file':
                output_filename = input("-> Enter filename to save decrypted content: ")
                with open(output_filename, "wb") as f:
                    f.write(decrypted_bytes)
                print("\n--- Verification ---")
                print(f"Decrypted file content saved to '{os.path.abspath(output_filename)}'")
        else:
            print("\n[!] Verification Failed: Could not decrypt the message.")

    except Exception as e:
        print(f"[!] An error occurred: {e}")
    finally:
        print("\n-> Closing sockets.")
        client.close()
        server.close()

if __name__ == "__main__":
    start_server()
