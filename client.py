# alice_client.py
import os
import socket
import struct
import subprocess

# --- Network Configuration ---
HOST = 'localhost'
PORT = 12345

def run_command(command, shell=True, check=True, input_data=None):
    """Helper to run shell commands, with optional stdin piping."""
    subprocess.run(command, shell=shell, check=check, input=input_data, text=True)

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

def get_user_votes():
    """Prompts the user to enter candidate names and votes."""
    print("\n--- Enter Vote Data ---")
    votes = []
    while True:
        candidate = input("Enter candidate name (or 'done' to finish): ")
        if candidate.lower() == 'done':
            break
        while True:
            try:
                num_votes = int(input(f"Enter votes for {candidate}: "))
                votes.append(f"{candidate},{num_votes}")
                break
            except ValueError:
                print("Invalid input. Please enter a number for votes.")
    return "\n".join(votes)

print("--- Alice (Client) is running ---")

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    print(f"📞 Connecting to Bob at {HOST}:{PORT}...")
    s.connect((HOST, PORT))
    print("🤝 Connected to Bob.")

    # Receive Bob's certificate
    bob_cert = "Alice/B.crt"
    receive_file(s, bob_cert)

    # Action 2: Alice verifies Bob and prepares the secure package
    print("\n[ALICE] Verifying Bob and preparing secure package...")
    run_command(f"openssl verify -CAfile CA/root.crt {bob_cert}")
    run_command("openssl rand -base64 32 > Alice/symkey.pem")
    run_command(f"openssl x509 -pubkey -noout -in {bob_cert} > Alice/pubkey-B.pem")
    run_command("openssl pkeyutl -encrypt -in Alice/symkey.pem -pubin -inkey Alice/pubkey-B.pem -out Alice/symkey.enc")
    
    print("[ALICE] Hashing and signing the symmetric key...")
    run_command("openssl dgst -sha1 -binary -out Alice/hash.bin Alice/symkey.pem")
    run_command("openssl pkeyutl -sign -in Alice/hash.bin -inkey Alice/privkey-A.pem -out Alice/signature.bin -pkeyopt digest:sha1")

    # Action 3: Alice sends her certificate and the package
    print("\n[ALICE] Sending certificate and encrypted key to Bob...")
    send_file(s, "Alice/A.crt")
    send_file(s, "Alice/symkey.enc")
    send_file(s, "Alice/signature.bin")

    # Action 5: Get user input and send encrypted data
    vote_data_string = get_user_votes()
    print("\n[ALICE] Encrypting your vote data and sending to Bob...")

    encrypt_command = "openssl enc -aes-256-cbc -pass file:Alice/symkey.pem -out Alice/votes_ciphertext.bin"
    run_command(encrypt_command, input_data=vote_data_string)

    send_file(s, "Alice/votes_ciphertext.bin")
    print("✅ Encrypted votes sent.")
    print("🎉 Alice's mission complete.")