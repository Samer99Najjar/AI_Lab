import random
from GeneticAlgo import GeneticAlgo


def genetic(k):
    if k == 6:
        ga = GeneticAlgo(k, 24, 30, 5000)
        ga.run()
    else:
        ga = GeneticAlgo(k, 122, 130, 5000)
        ga.run()


def generate_array(size):
    arr = []
    for j in range(size):
        arr.append(j)
    random.shuffle(arr)
    return arr


def checkArr(SOL, arr):
    for k in range(0, len(SOL), 2):
        i1 = arr[SOL[k]]
        i2 = arr[SOL[k + 1]]
        if i1 > i2:
            temp = arr[SOL[k]]
            arr[SOL[k]] = arr[SOL[k + 1]]
            arr[SOL[k + 1]] = temp
    for k in range(len(arr) - 1):
        if arr[k] > arr[k + 1]:
            return False
    return True


# we used this function to check if the answer is optimal
def check_sol(SOL, size):
    arrs = []
    for _ in range(1000):
        arrs.append(generate_array(size))

    for arr in arrs:
        print(checkArr(SOL, arr))


if __name__ == '__main__':
    running = True
    print('#########################################################################')
    print('LAB 4')
    print('MORSY BIADSY ID:318241221 \t SAMER NAJJAR ID:207477522')
    print('#########################################################################')
    while running:
        print()
        print("CHOOSE 1 OR 2 TO RUN:")
        input_file = input('1 --> K = 6 \t 2 --> K = 16 \t PRESS ANY OTHER KEY TO EXIT\n')
        if input_file == '1':
            genetic(6)
        else:
            if input_file == '2':
                genetic(16)
            else:
                running = False
    # check_sol(SOL, 6)
