
# encoding: utf-8
import os
c = 0.8
for i in range(0,5):
    c = c + 0.1
    file_data = ""
    f = open("tri.sh")
    lines = f.readlines()
    print(c)
    with open("tri.sh", "w") as fw:
        for line in lines:
            print(line)
            if "ir_w" in line:
                line = "--ir_w " + str(c) + " \\" '\n'
            if "logs-dir" in line:
                line = "--logs-dir ./0.5_rgb_" + str(c) + '_ir \\' + '\n'
            file_data += line
        fw.write(file_data)
    os.system('sh tri.sh')

