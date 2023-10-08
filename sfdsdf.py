file = open("./ed.txt", "r")

print("hey")

min = 1000
case = 0
case2 = 0
for line in file.readlines():
    list = [int(i) for i in line.split()]
    for i in list:
        if i < min and i !=0:
            min = i
            case2 = case
    case +=1


print(min)
print(case2)

file.close()