import os
import json
from pathlib import Path

def rename_json_files(root_dir: str = "extracted_cancer_cases"):
    """
    Rename JSON files under the extracted_cancer_cases folder.
    Format: subfolder_case_0X.json
    """
    
    # Check if the root directory exists
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist")
        return
    
    # Iterate through all subfolders in the root directory
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        
        # Check if it is a directory
        if not os.path.isdir(subfolder_path):
            continue
            
        print(f"Processing subfolder: {subfolder}")
        
        # Get all JSON files in this subfolder
        json_files = []
        for file in os.listdir(subfolder_path):
            if file.endswith('.json'):
                json_files.append(file)
        
        # Sort by filename to ensure consistent renaming order
        json_files.sort()
        
        # Rename the JSON files
        for index, json_file in enumerate(json_files, 1):
            old_path = os.path.join(subfolder_path, json_file)
            
            # Generate new filename: subfolder_case_0X.json
            new_filename = f"{subfolder}_case_{index:02d}.json"
            new_path = os.path.join(subfolder_path, new_filename)
            
            # Check if the new filename already exists
            if os.path.exists(new_path) and old_path != new_path:
                print(f"  Warning: File '{new_filename}' already exists, skipping rename of '{json_file}'")
                continue
            
            try:
                # Rename the file
                os.rename(old_path, new_path)
                print(f"  Renaming: {json_file} -> {new_filename}")
            except Exception as e:
                print(f"  Error: Failed to rename '{json_file}': {e}")
    
    print("Renaming complete!")

def preview_rename(root_dir: str = "extracted_cancer_cases"):
    """
    Preview the rename operation without actually renaming the files.
    """
    
    if not os.path.exists(root_dir):
        print(f"Error: Directory '{root_dir}' does not exist")
        return
    
    print("Previewing rename operation:")
    print("=" * 50)
    
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        
        if not os.path.isdir(subfolder_path):
            continue
            
        print(f"\nSubfolder: {subfolder}")
        
        json_files = []
        for file in os.listdir(subfolder_path):
            if file.endswith('.json'):
                json_files.append(file)
        
        json_files.sort()
        
        if not json_files:
            print("  No JSON files found")
            continue
        
        for index, json_file in enumerate(json_files, 1):
            new_filename = f"{subfolder}_case_{index:02d}.json"
            print(f"  {json_file} -> {new_filename}")

def main():
    """Main function"""
    print("JSON File Renaming Tool")
    print("=" * 30)
    
    # Check if the extracted_cancer_cases folder exists in the current directory
    root_dir = "extracted_cancer_cases"
    
    if not os.path.exists(root_dir):
        print(f"Could not find the '{root_dir}' folder in the current directory")
        print("Please ensure the script is in the same directory as the extracted_cancer_cases folder")
        return
    
    # Show preview
    print("1. Preview Rename Operation")
    preview_rename(root_dir)
    
    print("\n" + "=" * 50)
    
    # Ask the user whether to proceed
    while True:
        choice = input("\nDo you want to perform the rename operation? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            print("\nStarting the renaming process...")
            rename_json_files(root_dir)
            break
        elif choice in ['n', 'no']:
            print("Operation cancelled")
            break
        else:
            print("Please enter y or n")

if __name__ == "__main__":
    main()