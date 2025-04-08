import os
import zipfile

def zip_folder(folder_path, zip_name):
    zip_path = os.path.join(os.path.dirname(folder_path), zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                arcname = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname)
    print(f"✅ Zipped folder to: {zip_path}")

if __name__ == "__main__":
    folder_to_zip = "results"  # Replace with your folder name
    output_zip_name = "my_results.zip"
    zip_folder(folder_to_zip, output_zip_name)
