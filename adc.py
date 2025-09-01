import socket

# Additive cipher encryption
def additive_encrypt(plaintext, key):
    result = ""
    for char in plaintext:
        if char.isalpha():
            shift = 65 if char.isupper() else 97
            result += chr((ord(char) - shift + key) % 26 + shift)
        else:
            result += char
    return result

def start_client(host="127.0.0.1", port=65432):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # Step 1: Send the key to the server
    key = int(input("Enter encryption key (number): "))
    client_socket.sendall(str(key).encode())

    # Step 2: Send messages encrypted with the key
    while True:
        msg = input("Enter message (or 'exit' to quit): ")
        if msg.lower() == "exit":
            break

        encrypted = additive_encrypt(msg, key)
        print(f" Sending encrypted: {encrypted}")
        client_socket.sendall(encrypted.encode())

    client_socket.close()

if __name__ == "__main__":
    start_client()
