
def write(filename, content):
    with open(filename, 'w') as file_object:
        file_object.write(content)


def appendWrite(filename, content):
    with open(filename, 'a') as file_object:
        file_object.write(content)

