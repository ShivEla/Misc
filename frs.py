import socket
import os
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP

def decrypt_data(ciphertext, private_key):
    try:
        cipher = PKCS1_OAEP.new(private_key)
        decrypted_bytes = cipher.decrypt(ciphertext)
        return decrypted_bytes
    except (ValueError, TypeError) as e:
        print(f"[!] Decryption failed: {e}")
        return None

def start_server():
    host, port = '127.0.0.1', 9999
    print("RSA Server")

    # --- Step 1: Generate RSA Keys ---
    key_pair = RSA.generate(2048)
    public_key = key_pair.publickey()
    private_key = key_pair
    print(" Generated new 2048-bit RSA key pair for this session.")

    # --- Step 2: Set up Server Socket ---
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f" Server listening on {host}:{port}")

    client_socket, addr = server_socket.accept()
    print(f"\n Accepted connection from {addr}")

    try:
        # --- Step 3: Key Exchange and Data Reception ---
        # Send public key to the client
        client_socket.send(public_key.export_key())
        print(" Public key sent to client.")

        # First, receive the type of data ('TEXT' or 'FILE')
        data_type = client_socket.recv(1024).decode('utf-8')
        print(f" Client is sending data of type: {data_type}")

        # Then, receive the encrypted data
        encrypted_data = client_socket.recv(4096) # Buffer size for encrypted data
        print(f" Received {len(encrypted_data)} bytes of encrypted data.")
        print(f" Encrypted data: {encrypted_data.hex()[:80]}...")

        # --- Step 4: Decryption and Verification ---
        print("\n--- Decryption Process ---")
        decrypted_bytes = decrypt_data(encrypted_data, private_key)

        if decrypted_bytes:
            print("Decryption successful!")
            
            # --- Step 5: Handle Decrypted Data Based on Type ---
            if data_type == 'TEXT':
                # Decode bytes to a string for display
                decrypted_message = decrypted_bytes.decode('utf-8')
                print("\n--- Verification ---")
                print(f"Decrypted Message: {decrypted_message}")

            elif data_type == 'FILE':
                print("\n--- File Content Received ---")
                # Try to print a small preview, safely
                try:
                    print(f"Decrypted Preview (first 100 bytes): {decrypted_bytes[:100].decode('utf-8')}")
                except UnicodeDecodeError:
                    print(f"Decrypted Preview (first 100 bytes, hex): {decrypted_bytes[:100].hex()}")
                
                # Save the decrypted content to a file
                output_filename = input("\nEnter filename to save the decrypted content: ")
                with open(output_filename, "wb") as f:
                    f.write(decrypted_bytes)
                print(f"\n[+] VERIFICATION: Decrypted content successfully saved to '{os.path.abspath(output_filename)}'")
        else:
            print("[!] Could not decrypt the received data. It may be corrupt.")

    except Exception as e:
        print(f"\n[!] An error occurred: {e}")
    finally:
        print("\n Closing all sockets.")
        client_socket.close()
        server_socket.close()

if __name__ == "__main__":
    start_server()
