from pyuul import utils
import os
import torch
import shutil
#

def get_boundary(pdb_path):
    pdb_file = pdb_path
    print(pdb_file)
    desired_atom_name = 'DUM'
    z_coord = []

    with open(pdb_file, 'r') as file:
        # print(file)
        for line in file:
            # print(line)

            if ' DUM' in line:
                try:
                    z_coord.append(float(line[46:54]))
                except ValueError:
                    continue


        return min(z_coord), max(z_coord)




# def create_modified_pdb(pdb_path, output_folder, margin = 3.00):
#     margin = margin
#     min_z_boundary, max_z_boundary = get_boundary(pdb_path)
#     if min_z_boundary is None or max_z_boundary is None :
#         print(f"Unable to determine z_boundary for file: {pdb_path}")
#         return
#
#     min_z_threshold = min_z_boundary -margin
#     max_z_threshold = max_z_boundary + margin
#
#     try:
#         with open(pdb_path, 'r', encoding='ascii') as file:
#             lines = file.readlines()
#             modified_lines = []
#             try:
#                 for line in lines:
#                     print(line)
#                     if line.startswith('ATOM'):
#
#                         z = float(line[47:54])
#
#                         try:
#
#                             if z <= max_z_threshold and z >= min_z_threshold:
#
#                                 modified_lines.append(line)
#                             else:
#                                 continue  # Keep the line as is
#                         except ValueError:
#                             print(f"Error converting value to float: {pdb_path}")
#                             continue
#                     else:
#                         modified_lines.append(line)  # Keep non-'ATOM' lines as is
#
#             except UnicodeDecodeError:
#                 print(f"Error decoding file: {pdb_path}")
#                 return
#         return modified_lines
#
#     except :
#         continue
#     # with open(output_folder, 'w', encoding='ascii') as file:
#     #     file.writelines(modified_lines)
def create_modified_pdb(pdb_path, output_folder, margin=3.0):
    min_z_boundary, max_z_boundary = get_boundary(pdb_path)
    if min_z_boundary is None or max_z_boundary is None:
        print(f"Unable to determine z_boundary for file: {pdb_path}")
        return
    count=0
    min_z_threshold = min_z_boundary - margin
    max_z_threshold = max_z_boundary + margin

    try:
        with open(pdb_path, 'r', encoding='ascii') as file:
            lines = file.readlines()
            modified_lines = []

            for line in lines:
                if line.startswith('ATOM'):
                    # print(line)
                    z = float(line[46:54])

                    if z <= max_z_threshold and z >= min_z_threshold:
                        count +=1
                        modified_lines.append(line)
                        # print(count)
                    else:
                        continue
                else:
                    # modified_lines.append(line)
                    continue

        if count == 0:
            print("No ATOM lines satisfied the Z coordinate condition. Skipping file modification.")
            return None

        with open(output_folder, 'w', encoding='ascii') as file:
            file.writelines(modified_lines)
        # return modified_lines

        print("Modified PDB file created successfully.")
    except (UnicodeDecodeError, ValueError, IOError) as e:
        print(f"Error processing file: {pdb_path}")
        print(f"Error message: {str(e)}")

# Usage example
# create_modified_pdb("input.pdb", "output.pdb", margin=2.0)

#


def create_Z_trans_folder(trans_folder_path,output_folder):
    try:
        if not os.path.exists(output_folder):
            print('No directory making it !!')
            os.mkdir(output_folder)

        if os.path.exists(output_folder):
            print('Directory there, deleting and making new')
            shutil.rmtree(output_folder)
            os.mkdir(output_folder)
    except OSError:
        print(f'Error: Could not create a destination folder {output_folder}')
        exit()
    #create list of files in a folder
    file_names = os.listdir(trans_folder_path)
    # Filter out files starting with a dot
    files = [file for file in file_names if not file.startswith(".")]
    for file_name in files:
        file_path = os.path.join(trans_folder_path, file_name)
        output_file_path = os.path.join(output_folder,file_name)
        # print(file_path)
        create_modified_pdb(file_path,output_file_path,margin=3.000)
        # print(get_boundary(file_path))




# coords, radius, at_channel=get_coords('/Users/bivekpokhrel/PycharmProjects/database/data/new_z_trans')
protien_path='/Users/bivekpokhrel/PycharmProjects/database/data/new_z_trans'
protien_path2='/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder'
protien_path3='/Users/bivekpokhrel/PycharmProjects/database/data/new_z_trans/2wwb.pdb'
protein_path4='/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/3hfx.pdb'
# create_Z_trans_folder(protien_path2,protien_path)
# coords=utils.parsePDB(protien_path, keep_hetatm=False)
# # print(coords)
# print(get_boundary('/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/5aji.pdb'))
#
# # print(create_modified_pdb('/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/5aji.pdb','/Users/bivekpokhrel/PycharmProjects/database/data/Z_trans_folder'))
# # print(get_boundary(protien_path2))
# create_modified_pdb('/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/5aji.pdb','/Users/bivekpokhrel/PycharmProjects/database/data/Z_trans_folder')




# print(get_boundary('/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/3j7q.pdb'))
# coords,_=parsePDB(protien_path3)
# print(len(coords))
# print(coords[0].shape)
def get_coords(protien_path):
    coords,atname=utils.parsePDB(protien_path, keep_hetatm=False)[0],utils.parsePDB(protien_path, keep_hetatm=False)[1]
    radius=utils.atomlistToRadius(atname)
    atom_channel= utils.atomlistToChannels(atname)
    return coords, radius, atom_channel
# for fil in sorted(os.listdir(protien_path)):
#     file=os.path.join(protien_path,fil)
#     print(file)
#     coords, radius, atom_channel = get_coords(file)
#
#     print(len(coords))
#     print(coords[0].shape)
#     # print(coords)

# print(get_coords(protien_path))
# coords, _ = utils.parsePDB('/Users/bivekpokhrel/PycharmProjects/database/data/new_z_trans/3j7q.pdb')
# print(coords)
# print(create_modified_pdb('/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/3j7q.pdb','/Users/bivekpokhrel/PycharmProjects/database/data/Z_trans_folder'))