
from pathlib import Path

FILE_TYPES = {
    "plots": [".png", ".jpg", ".jpeg"],
    "models": [".pkl"]
}

def organizeFiles(path):
    # Make new directories
    for key in FILE_TYPES.keys():
        upPath = Path(path).parent
        newDir = Path.joinpath(upPath, key)
        newDir.mkdir(exist_ok=True)
    newDir = Path.joinpath(upPath, "Others")
    newDir.mkdir(exist_ok=True)
    
    # Organize files into directories
    for file in Path.iterdir(path):
        if Path.is_file(file):
            type = file.suffix
            dirName = None
            for k, v in FILE_TYPES.items():
                if type in v:
                    dirName = k
                    break
            if dirName is None:
                continue
            dest_dir = upPath / dirName
            dest = dest_dir / file.name
            file.rename(dest)

            print(f"{file} moved to {dest_dir}.")

def main():
    PATH = Path(r"notebooks")
    organizeFiles(PATH)



if __name__ == "__main__":
    main()




