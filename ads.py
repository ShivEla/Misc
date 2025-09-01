import socket

# Additive cipher decryption
def additive_decrypt(ciphertext, key):
    result = ""
    for char in ciphertext:
        if char.isalpha():
            shift = 65 if char.isupper() else 97
            result += chr((ord(char) - shift - key) % 26 + shift)
        else:
            result += char
    return result

def start_server(host="127.0.0.1", port=65432):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)

    print(f"Server running on {host}:{port}... Waiting for client.")

    conn, addr = server_socket.accept()
    print(f"Connected by {addr}")

    # First, receive the key from the client
    key_data = conn.recv(1024).decode()
    key = int(key_data)
    print(f"Key received from client: {key}")

    while True:
        data = conn.recv(1024).decode()
        if not data:
            break
        print(f" Ciphertext received: {data}")
        decrypted = additive_decrypt(data, key)
        print(f"Decrypted plaintext: {decrypted}")

    conn.close()
    server_socket.close()

if __name__ == "__main__":
    start_server()
