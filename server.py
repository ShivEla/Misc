# bob_server.py
import os
import socket
import struct
import subprocess

# --- Network Configuration ---
HOST = 'localhost'
PORT = 12345

def run_command(command, shell=True, check=True):
    """Helper to run shell commands."""
    subprocess.run(command, shell=shell, check=check)

def send_file(sock, filepath):
    """Sends a file over a socket by first sending its size."""
    filesize = os.path.getsize(filepath)
    print(f"📦 Sending {os.path.basename(filepath)} ({filesize} bytes)...")
    sock.sendall(struct.pack('>Q', filesize))
    with open(filepath, 'rb') as f:
        while read_bytes := f.read(1024):
            sock.sendall(read_bytes)
    print(f"✔️ Sent {os.path.basename(filepath)}.")

# --- MODIFIED FUNCTION START ---
def receive_file(sock, save_path):
    """Receives a file from a socket and saves it robustly."""
    packed_size = sock.recv(struct.calcsize('>Q'))
    if not packed_size:
        return None
    filesize = struct.unpack('>Q', packed_size)[0]
    print(f"⏳ Receiving {os.path.basename(save_path)} ({filesize} bytes)...")

    with open(save_path, 'wb') as f:
        bytes_received = 0
        while bytes_received < filesize:
            # Calculate how many bytes we still need to receive
            remaining = filesize - bytes_received
            # Receive up to 1024 bytes, or the remaining amount if smaller
            chunk = sock.recv(min(1024, remaining))
            if not chunk:
                break
            f.write(chunk)
            bytes_received += len(chunk)
    print(f"✔️ Received and saved to {os.path.basename(save_path)}.")
    return save_path
# --- MODIFIED FUNCTION END ---


print("--- Bob (Server) is running ---")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"👂 Bob is listening on {HOST}:{PORT}")
    conn, addr = s.accept()
    with conn:
        print(f"🤝 Connected by {addr}")

        # Action 1: Bob sends his certificate
        print("\n[BOB] Sending my certificate to Alice...")
        send_file(conn, "Bob/B.crt")

        # Wait for Alice's package
        print("\n[BOB] Waiting for Alice's package...")
        receive_file(conn, "Bob/A.crt")
        receive_file(conn, "Bob/symkey.enc")
        receive_file(conn, "Bob/signature.bin")

        # Action 4: Bob verifies Alice and decrypts the key
        print("\n[BOB] Received package. Verifying Alice and decrypting key...")
        run_command("openssl verify -CAfile CA/root.crt Bob/A.crt")
        run_command("openssl pkeyutl -decrypt -in Bob/symkey.enc -inkey Bob/privkey-B.pem -out Bob/symkey.pem")
        run_command("openssl x509 -pubkey -noout -in Bob/A.crt > Bob/pubkey-A.pem")
        
        print("[BOB] Hashing symmetric key and verifying signature...")
        run_command("openssl dgst -sha1 -binary -out Bob/hash.bin Bob/symkey.pem")
        run_command("openssl pkeyutl -verify -in Bob/hash.bin -sigfile Bob/signature.bin -inkey Bob/pubkey-A.pem -pubin -pkeyopt digest:sha1")
        
        print("✅ Secure key exchange complete!")

        # Wait for the encrypted data
        print("\n[BOB] Waiting for encrypted vote data...")
        receive_file(conn, "Bob/votes_ciphertext.bin")

        # Action 6: Bob decrypts the data
        print("\n[BOB] Received encrypted data. Decrypting...")
        run_command("openssl enc -aes-256-cbc -d -pass file:Bob/symkey.pem -in Bob/votes_ciphertext.bin -out Bob/decrypted_votes.txt")
        print("✅ Votes decrypted to Bob/decrypted_votes.txt")

        print("\n--- DECRYPTED VOTE DATA ---")
        with open("Bob/decrypted_votes.txt", "r") as f:
            print(f.read())
        print("--------------------------")
        print("🎉 Bob's mission complete.")