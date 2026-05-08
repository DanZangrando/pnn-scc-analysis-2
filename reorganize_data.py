import os
import shutil

raw_dir = 'data/raw'
groups = {
    'NONE_MACHO': ['43', '45', '63', '62'],
    'NONE_HEMBRA': ['51', '53', '47', '83'],
    'ACF_14_DIAS_MACHO': ['41', '42', '46', '64'],
    'ACF_14_DIAS_HEMBRA': ['49', '50', '55', '71']
}

# Create group directories
for group in groups:
    path = os.path.join(raw_dir, group)
    if not os.path.exists(path):
        os.makedirs(path)

# Move files
files = [f for f in os.listdir(raw_dir) if f.endswith('.czi')]
for file in files:
    # Example filename: ACF_43_SSC_C_2_AGR488_WFA647_PV546_DAPI405_10x_E.czi
    # We try to find the ID in the filename
    parts = file.split('_')
    file_id = None
    if len(parts) > 1:
        file_id = parts[1]
    
    if file_id:
        moved = False
        for group, ids in groups.items():
            if file_id in ids:
                src = os.path.join(raw_dir, file)
                dst = os.path.join(raw_dir, group, file)
                print(f"Moving {file} to {group}")
                shutil.move(src, dst)
                moved = True
                break
        if not moved:
            print(f"No group found for {file} (ID: {file_id})")
    else:
        print(f"Could not extract ID from {file}")
