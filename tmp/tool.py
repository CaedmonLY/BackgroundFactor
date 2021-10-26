import os
def makeTest(path="9"):
    t=os.listdir(path)
    file=open("test.txt",mode='w')
    for i in t:
        file.writelines(i)
        file.writelines('\n')
    pass


if __name__ == '__main__':
    makeTest()