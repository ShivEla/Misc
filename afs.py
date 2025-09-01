import socket
from math import gcd

# Find modular multiplicative inverse of key mod 26
def mod_inverse(a, m=26):
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None

# Affine cipher decryption
def affine_decrypt(ciphertext, a, b):
    a_inv = mod_inverse(a, 26)
    if a_inv is None:
        return "❌ Invalid multiplicative key (no modular inverse)"
    
    result = ""
    for char in ciphertext:
        if char.isalpha():
            shift = 65 if char.isupper() else 97
            result += chr(((a_inv * ((ord(char) - shift - b)) % 26) + shift))
        else:
            result += char
    return result

def start_server(host="127.0.0.1", port=65432):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Affine Cipher Server running on {host}:{port}... Waiting for client.")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    # Receive keys
    keys = conn.recv(1024).decode().split(",")
    a = int(keys[0])
    b = int(keys[1])
    print(f" Keys received from client: a={a}, b={b}")

    while True:
        data = conn.recv(1024).decode()
        if not data:
            break
        print(f"Ciphertext received: {data}")
        decrypted = affine_decrypt(data, a, b)
        print(f" Decrypted plaintext: {decrypted}")

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    start_server()
