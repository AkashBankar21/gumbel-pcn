import zipfile
import os

def extract_zip(zip_path, extract_to):
    """
    Extracts a ZIP file to a specified directory.
    
    Args:
        zip_path (str): Path to the ZIP file.
        extract_to (str): Directory to extract the contents to.
    """
    if not zipfile.is_zipfile(zip_path):
        print(f"{zip_path} is not a valid ZIP file.")
        return
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(f"Extracting {zip_path} to {extract_to}...")
        zip_ref.extractall(extract_to)
        print("Extraction complete.")

# Example usage
zip_file_path = 'ShapeNet55.zip'       # Replace with your ZIP file path
destination_folder = 'extracted/'  # Replace with your desired output folder

# Create the output folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

extract_zip(zip_file_path, destination_folder)
