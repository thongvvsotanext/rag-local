# Make the script executable

chmod +x generate_secret_key.sh

# Basic usage - update settings.yml

./generate_secret_key.sh

# Just generate and display a key

./generate_secret_key.sh --generate-only

# Generate longer key (128 characters)

./generate_secret_key.sh -l 128

# Update specific file with verbose output

./generate_secret_key.sh -f config.yml --verbose

# Generate without creating backup

./generate_secret_key.sh --no-backup
