# setup_pki.py (Recommended Minor Change)
import os
import shutil
import subprocess
import sys

def run_command(command, shell=True):
    """A simple helper to run shell commands and exit on error."""
    try:
        subprocess.run(command, shell=shell, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}\n{e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)

print("--- Initializing Workspace ---")
# "shared_network" is no longer needed, so it has been removed from this list.
for dirname in ["CA", "Alice", "Bob"]:
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)
print("✅ Workspace cleaned and directories created.\n")

print("--- Pre-computation: Creating PKI ---")
# Create CA
print("[CA] Generating root key and self-signed certificate...")
run_command("openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out CA/rootkey.pem")
run_command('openssl req -x509 -new -nodes -key CA/rootkey.pem -sha256 -days 1024 -out CA/root.crt -subj "/C=IN/ST=TamilNadu/L=Coimbatore/O=MyCA/CN=myca.com"')

# Create Alice's Certificate
print("[ALICE] Generating key and signing request...")
run_command("openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out Alice/privkey-A.pem")
run_command('openssl req -new -key Alice/privkey-A.pem -out Alice/A-req.csr -subj "/C=IN/ST=TamilNadu/L=Coimbatore/O=AliceCorp/CN=alice.com"')
print("[CA] Signing Alice's certificate...")
run_command("openssl x509 -req -in Alice/A-req.csr -CA CA/root.crt -CAkey CA/rootkey.pem -CAcreateserial -out Alice/A.crt -days 500 -sha256")

# Create Bob's Certificate
print("[BOB] Generating key and signing request...")
run_command("openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out Bob/privkey-B.pem")
run_command('openssl req -new -key Bob/privkey-B.pem -out Bob/B-req.csr -subj "/C=IN/ST=TamilNadu/L=Coimbatore/O=BobCorp/CN=bob.com"')
print("[CA] Signing Bob's certificate...")
run_command("openssl x509 -req -in Bob/B-req.csr -CA CA/root.crt -CAkey CA/rootkey.pem -out Bob/B.crt -days 500 -sha256")

print("\n✅ PKI setup complete. Ready to run client and server.")