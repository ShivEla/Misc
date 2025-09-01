import socket
from math import gcd

# Affine cipher encryption
def affine_encrypt(plaintext, a, b):
    result = ""
    for char in plaintext:
        if char.isalpha():
            shift = 65 if char.isupper() else 97
            result += chr(((a * (ord(char) - shift) + b) % 26) + shift)
        else:
            result += char
    return result

def start_client(host="127.0.0.1", port=65432):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Step 1: Get keys
    while True:
        a = int(input("Enter multiplicative key (coprime with 26): "))
        if gcd(a, 26) == 1:
            break
        else:
            print("Invalid key! 'a' must be coprime with 26.")
    b = int(input("Enter additive key (0-25): "))

    # Send keys to server
    client_socket.sendall(f"{a},{b}".encode())

    # Step 2: Encrypt messages
    while True:
        msg = input("Enter message (or 'exit' to quit): ")
        if msg.lower() == "exit":
            break

        encrypted = affine_encrypt(msg, a, b)
        print(f" Sending encrypted: {encrypted}")
        client_socket.sendall(encrypted.encode())

    client_socket.close()

if __name__ == "__main__":
    start_client()
