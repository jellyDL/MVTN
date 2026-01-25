import os

if __name__ == "__main__":
    print("This is a script to create teeth dataset.")
    raw_data_path = "/home/jelly/Datasets/Crown_Pair_Dataset_2026_0105"
    new_data_path = "../data_teeth/TeethDataset"
    os.makedirs(new_data_path, exist_ok=True)
    
    raw_cad_path = os.path.join(raw_data_path, "CAD")
    raw_scan_path = os.path.join(raw_data_path, "SCAN")
    cad_files = sorted(os.listdir(raw_cad_path))

    for iter, cad_file in enumerate(cad_files):
        print(f"Processing {cad_file}  {iter+1}/{len(cad_files)}")
        if cad_file.endswith(".stl"):
            cad_file_path = os.path.join(raw_cad_path, cad_file)
            scan_file_path = os.path.join(raw_scan_path, cad_file)
            
            new_folder = os.path.join(new_data_path, f"{iter:04d}")
            os.makedirs(new_folder, exist_ok=True)
            new_train_path = os.path.join(new_folder, "train")
            os.makedirs(new_train_path, exist_ok=True)
            new_test_path = os.path.join(new_folder, "test")
            os.makedirs(new_test_path, exist_ok=True)
            cmd = f"cp {cad_file_path} {new_train_path}"
            os.system(cmd)
            cmd = f"cp {scan_file_path} {new_test_path}"
            os.system(cmd)
            
    
    num = len(cad_files)
    print(f"Total {num} samples processed.")
    train_faults_file = os.path.join(new_data_path, "..", "train_faults.txt")
    with open(os.path.join(train_faults_file), "w") as f:
        for i in range(num):
            f.write(f"{i}\n")
            
    test_faults_file = os.path.join(new_data_path, "..", "test_faults.txt")
    with open(os.path.join(test_faults_file), "w") as f:
        for i in range(num):
            f.write(f"{i}\n")
    
    
    
    print("Teeth dataset creation completed.")