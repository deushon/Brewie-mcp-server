# ⚠️ SECURITY WARNING

## REAL TRANSFERS IN MAINNET

**WARNING**: This tool performs **REAL** SOL transfers in the main Solana network (mainnet). 

### 🚨 CRITICALLY IMPORTANT:

1. **Use only test amounts** - start with 0.001 SOL
2. **Verify recipient address** - make sure the QR code contains the correct address
3. **Keep private key secure** - the `master_sh/sol_private_key` file should have 600 permissions
4. **Create a backup** - save the private key in a secure location
5. **Use a separate wallet** - don't use your main wallet for testing

### 🔒 Security recommendations:

- Set file permissions for the key file: `chmod 600 master_sh/sol_private_key`
- Regularly check wallet balance
- Keep a log of all transactions
- Use VPN when working with wallet
- Never share your private key

### 📋 Before testing:

1. Make sure the wallet has enough SOL for transfer + fee (~0.000005 SOL)
2. Verify that the QR code contains a valid Solana address
3. Start with minimum amount (0.001 SOL)
4. Check the result in Solana Explorer

### 🆘 In case of problems:

- Check console logs
- Make sure RPC endpoint is accessible
- Check wallet balance
- In case of error, transaction may be cancelled

**REMEMBER: All transfers are irreversible!**