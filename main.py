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


    print("Getting Database")
    directories = getDatabase.main()

    print("Training Model")
    trainModel.main(directories)

    print("Finished")





if __name__ == "__main__":
    main()
