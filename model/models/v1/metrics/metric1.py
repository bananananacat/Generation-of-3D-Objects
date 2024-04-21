def calc(predict, target, eps):
    n = len(target) 
    sum = 0
    for i in predict:
        flag = 0
        for j in target:
            if (i.x - j.x) ** 2 + (i.y - j.y) ** 2 + (i.z - j.z) ** 2 < eps:
                flag = 1
            sum += flag
    return sum / n
