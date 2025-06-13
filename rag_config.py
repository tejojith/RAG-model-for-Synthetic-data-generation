import os

def check_for_file():
    def is_faiss_folder(path):
        """Check if a folder contains FAISS files"""
        if not os.path.isdir(path):
            return False
        return os.path.exists(os.path.join(path, "index.faiss")) and os.path.exists(os.path.join(path, "index.pkl"))

    def find_faiss_dirs(base_dir):
        faiss_dirs = []
        for root, dirs, files in os.walk(base_dir):
            for d in dirs:
                full_path = os.path.join(root, d)
                if is_faiss_folder(full_path):
                    faiss_dirs.append(full_path)
        return faiss_dirs

    directory_to_search = '.'  # or specify another base directory
    found_dirs = find_faiss_dirs(directory_to_search)

    def create_new_faiss_dir():
        print("Creating a new FAISS vector store directory.")
        name = input("Enter the name of the new FAISS DB folder: ").strip()
        os.makedirs(name, exist_ok=True)
        return name

    if found_dirs:
        print(f"Found existing FAISS vector DB: {found_dirs[0]}")
        use_existing = input("Do you want to continue with the existing one? (y/n): ").strip().lower()
        if use_existing == 'y':
            DB_PATH = found_dirs[0]
        else:
            DB_PATH = create_new_faiss_dir()
    else:
        print("No existing FAISS DB found.")
        DB_PATH = create_new_faiss_dir()

    PROJECT_PATH = "database"
    return PROJECT_PATH, DB_PATH
