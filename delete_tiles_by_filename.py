import os


def delete_files(directory_path, filename):
    for file in os.listdir(directory_path):
        # print(file)
        if (file.find(filename) != -1):
            print("Deleted", file)
            os.remove(directory_path + file)


if __name__ == "__main__":
    directory = '/home/cougarnet.uh.edu/pcicales/Desktop/Chang_set_1/syed_delete_files_practice/'
    delete_files(directory, "syed_test")  # Will delete files in specified directory that contain 'syed_test'
    # somewhere in their filename
