import socket
from math import gcd

# Find modular multiplicative inverse of key mod 26
def mod_inverse(key, m=26):
    for x in range(1, m):
        if (key * x) % m == 1:
            return x
    return None

# Multiplicative cipher decryption
def multiplicative_decrypt(ciphertext, key):
    inv_key = mod_inverse(key, 26)
    if inv_key is None:
        return " Invalid key (no modular inverse)"
    
    result = ""
    for char in ciphertext:
        if char.isalpha():
            shift = 65 if char.isupper() else 97
            result += chr(((ord(char) - shift) * inv_key) % 26 + shift)
        else:
            result += char
    return result

def start_server(host="127.0.0.1", port=65432):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Multiplicative Cipher Server running on {host}:{port}... Waiting for client.")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    # First, receive the key
    key_data = conn.recv(1024).decode()
    key = int(key_data)
    print(f" Key received from client: {key}")

    while True:
        data = conn.recv(1024).decode()
        if not data:
            break
        print(f"Ciphertext received: {data}")
        decrypted = multiplicative_decrypt(data, key)
        print(f"Decrypted plaintext: {decrypted}")

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    start_server()
