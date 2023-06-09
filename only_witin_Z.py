# from pyuul import utils
import os
import torch
import shutil
#
# # Modified_trans_folder_path = "/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/1c17.pdb"
# output_folder='/Users/bivekpokhrel/PycharmProjects/database/data/new_z_trans/1c17.pdb'
def get_boundary(pdb_path):
    pdb_file = pdb_path
    print(pdb_file)
    desired_atom_name = 'DUM'
    z_coord=[]

    with open(pdb_file, 'r') as file:
        for line in file:


            if ' DUM' in line:
                z_coord.append(float(line[46:54]))
        return min(z_coord), max(z_coord)
            #
            #     columns = line.split()
            #     print(columns)
            #
            #     if columns[3] == desired_atom_name:
            #
            #         value = float(columns[7])
            #         if value is not None:
            #             return abs(value)
            #         else :
            #             return 100.00
            #     if columns[2] == desired_atom_name:
            #         value = float(columns[6])
            #         if value is not None:
            #             return abs(value)
            #         else :
            #             return 100.00


#

def create_modified_pdb(pdb_path, output_folder, margin = 3.00):
    margin = margin
    min_z_boundary, max_z_boundary = get_boundary(pdb_path)
    if min_z_boundary is None or max_z_boundary is None :
        print(f"Unable to determine z_boundary for file: {pdb_path}")
        return

    min_z_threshold = min_z_boundary -margin
    max_z_threshold = max_z_boundary + margin

    try:
        with open(pdb_path, 'r', encoding='ascii') as file:
            lines = file.readlines()
            modified_lines = []
            for line in lines:
                if line.startswith('ATOM'):
                    # columns = line.split()
                    z = float(line[47:54])
                    # print(z)
                    try:

                        if z <= max_z_threshold and z >= min_z_threshold:
                            # print(line)
                            modified_lines.append(line)
                        else:
                            continue  # Keep the line as is
                    except ValueError:
                        print(f"Error converting value to float: {pdb_path}")
                        continue
                else:
                    modified_lines.append(line)  # Keep non-'ATOM' lines as is

    except UnicodeDecodeError:
        print(f"Error decoding file: {pdb_path}")
        return
    # return modified_lines
    with open(output_folder, 'w', encoding='ascii') as file:
        file.writelines(modified_lines)


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


# print(coord.shape)

# create_modified_pdb('/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/4jr9.pdb',output_folder,margin=3.000)
#
# print(get_boundary('/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/6pzt.pdb'))
# # print(get_boundary('/Users/bivekpokhrel/PycharmProjects/database/data/new_z_trans/1mt5.pdb'))
# # my_parsePDB('/Users/bivekpokhrel/PycharmProjects/database/data/new_z_trans')


# def get_coords(protien_path):
#     coords,atname=utils.parsePDB(protien_path, keep_hetatm=False)[0],utils.parsePDB(protien_path, keep_hetatm=False)[1]
#     radius=utils.atomlistToRadius(atname)
#     atom_channel= utils.atomlistToChannels(atname)
#     return coords, radius, atom_channel

# coords, radius, at_channel=get_coords('/Users/bivekpokhrel/PycharmProjects/database/data/new_z_trans')
protien_path='/Users/bivekpokhrel/PycharmProjects/database/data/new_z_trans'
protien_path2='/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder'
protien_path3='/Users/bivekpokhrel/PycharmProjects/database/data/new_z_trans/2wwb.pdb'
protein_path4='/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/3hfx.pdb'
# coords=utils.parsePDB(protien_path, keep_hetatm=False)
# # print(coords)
# print(get_boundary('/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/5aji.pdb'))
#
# # print(create_modified_pdb('/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/5aji.pdb','/Users/bivekpokhrel/PycharmProjects/database/data/Z_trans_folder'))
# # print(get_boundary(protien_path2))
# create_modified_pdb('/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/5aji.pdb','/Users/bivekpokhrel/PycharmProjects/database/data/Z_trans_folder')


# def parsePDB(PDBFile,keep_only_chains=None,keep_hetatm=False,bb_only=False):
#     """
#     function to parse pdb files. It can be used to parse a single file or all the pdb files in a folder. In case a folder is given, the coordinates are gonna be padded
#
#     Parameters
#     ----------
#     PDBFile : str
#     path of the PDB file or of the folder containing multiple PDB files
#     bb_only : bool
#     if True ignores all the atoms but backbone N, C and CA
#     keep_only_chains : str or None
#     ignores all the chain but the one given. If None it keeps all chains
#     keep_hetatm : bool
#     if False it ignores heteroatoms
#     Returns
#     -------
#     coords : torch.Tensor
#     coordinates of the atoms in the pdb file(s). Shape ( batch, numberOfAtoms, 3)
#
#     atomNames : list
#     a list of the atom identifier. It encodes atom type, residue type, residue position and chain
#
#     """
#     PADDING_INDEX = 999.0
#     NONEXISTING_HASH = -1
#     MISSING_ATOM = -999
#     NON_ACCEPTABLE_ENERGY = 999
#     NON_ACCEPTABLE_DISTANCE = 999
#     MISSING_INDEX = -1
#
#     EPS = 0.00001
#
#     bbatoms = ["N", "CA", "C"]
#     if not os.path.isdir(PDBFile):
#         fil = PDBFile
#         coords = []
#         atomNames = []
#         cont = -1
#         oldres = -999
#         for line in open(fil).readlines():
#
#             if line[:4] == "ATOM":
#                 if keep_only_chains is not None and (not line[21] in keep_only_chains):
#                     continue
#                 if bb_only and not line[12:16].strip() in bbatoms:
#                     continue
#                 if oldres != int(line[22:26]):
#                     cont += 1
#                     oldres = int(line[22:26])
#                 resnum = int(line[22:26])
#                 atomNames += [line[17:20].strip() + "_" + str(resnum) + "_" + line[12:16].strip() + "_" + line[21]]
#
#                 x = float(line[30:38])
#                 y = float(line[38:46])
#                 z = float(line[47:54])
#                 coords += [[x, y, z]]
#
#             elif line[:6] == "HETATM" and keep_hetatm:
#
#                 resname_het = line[17:20].strip()
#                 resnum = int(line[22:26])
#                 x = float(line[30:38])
#                 y = float(line[38:46])
#                 z = float(line[47:54])
#                 coords += [[x, y, z]]
#                 atnameHet = line[12:16].strip()
#                 atomNames += [resname_het + "_" + str(resnum) + "_" + atnameHet + "_" + line[21]]
#         return torch.tensor(coords).unsqueeze(0), [atomNames]
#     else:
#         coords = []
#         atomNames = []
#
#         for fil in sorted(os.listdir(PDBFile)):
#
#             atomNamesTMP = []
#             coordsTMP = []
#             cont = -1
#             oldres = -999
#             for line in open(PDBFile + "/" + fil).readlines():
#
#                 if line[:4] == "ATOM":
#                     if keep_only_chains is not None and (not line[21] in keep_only_chains):
#                         continue
#                     if bb_only and not line[12:16].strip() in bbatoms:
#                         continue
#                     if oldres != int(line[22:26]):
#                         cont += 1
#                         oldres = int(line[22:26])
#
#                     resnum = int(line[22:26])
#                     atomNamesTMP += [
#                         line[17:20].strip() + "_" + str(resnum) + "_" + line[12:16].strip() + "_" + line[21]]
#
#                     x = float(line[30:38])
#                     y = float(line[38:46])
#                     z = float(line[47:54])
#                     coordsTMP += [[x, y, z]]
#
#                 elif line[:6] == "HETATM" and keep_hetatm:
#                     if line[17:20].strip() != "GTP":
#                         continue
#                     x = float(line[30:38])
#                     y = float(line[38:46])
#                     z = float(line[47:54])
#                     resnum = int(line[22:26])
#                     coordsTMP += [[x, y, z]]
#                     atnameHet = line[12:16].strip()
#                     atomNamesTMP += ["HET_" + str(resnum) + "_" + atnameHet + "_" + line[21]]
#             coords += [torch.tensor(coordsTMP)]
#             atomNames += [atomNamesTMP]
#         return coords

        # return torch.torch.nn.utils.rnn.pad_sequence(coords, batch_first=True, padding_value=PADDING_INDEX), atomNames
# print(get_boundary('/Users/bivekpokhrel/PycharmProjects/database/data/trans_folder/5aji.pdb'))
# coords,_=parsePDB(protien_path3)
# print(len(coords))
# print(coords[0].shape)

# for fil in sorted(os.listdir(protien_path)):
#     file=os.path.join(protien_path,fil)
#     coords= parsePDB(file)
#     print(file)
#     print(len(coords))
#     print(coords[0].shape)
#     # print(coords)