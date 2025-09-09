# SOL Transfer Setup Instructions

## ⚠️ WARNING: REAL TRANSFERS IN MAINNET

This tool performs **REAL** SOL transfers in the main Solana network (mainnet). 
Make sure you understand the risks and use only test amounts!

## Installing Dependencies

Install additional dependencies for working with QR codes and Solana:

```bash
pip install -r requirements_sol.txt
```

### Installation Check

Run a test to check all imports:

```bash
python3 test_imports.py
```

If all tests pass, the system is ready to work!

## Private Key Format

Create a file `master_sh/sol_private_key` with your Solana private key.

### File Format:
```
[your_private_key_in_base58_format]
```

### Example:
```
5KJvsngHeMpm884wtkJQQLi2Xr3q7D4z79dAMRzjXQvsRzVtQU9
```

### How to get private key:

1. **From Phantom Wallet:**
   - Open Phantom
   - Settings → Show Private Key
   - Copy the key in base58 format

2. **From Solflare:**
   - Settings → Export Private Key
   - Choose base58 format

3. **From Solana CLI command line:**
   ```bash
   solana-keygen new --outfile ~/my-wallet.json
   # Key will be in file in JSON array format
   ```

### Important:
- Never share your private key
- Keep the file in a secure location
- Make sure the file has correct permissions (600)

## Testing

After setup you can test the tool:

```python
# In Python console or through MCP client
BrewPay(0.001)  # Transfer 0.001 SOL (recommended to start with small amounts)
```

### Work Process:
1. Robot clears photo folder
2. Takes a photo
3. Searches for QR code with SOL wallet address
4. Validates address
5. Loads your private key
6. Creates and signs transaction
7. Sends transaction to Solana network
8. Waits for confirmation
9. Returns result with transaction signature

## Security

- Private key is stored locally in file `master_sh/sol_private_key`
- File should be readable only by owner
- Recommended to use separate wallet for testing