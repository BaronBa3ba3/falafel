import os
import subprocess

import getDatabase
import trainModel




def main():
#### Run script using subprocess

    # result = subprocess.run(['python', 'getDatabase.py'], capture_output=True, text=True)
    # # To capture output
    # print(result.stdout)
    # # To handle errors
    # if result.returncode != 0:
    #     print(f"Error: {result.stderr}")

    x = 4


    print("\n\t 1-Getting Database\n")
    directories = getDatabase.main()

    print("\n\t 2-Training Model\n")
    trainModel.main(directories)

    print("\n\t 3-Finished\n")





if __name__ == "__main__":
    main()
