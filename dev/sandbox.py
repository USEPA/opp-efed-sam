a = '7.0'
b = '10W'

def p(num):
    return str(int(float(num))).zfill(2)

print(p(a))
print(p(b))