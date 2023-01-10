regions = ['7.0', '10U', '30']
print('7.0'.isdigit())
regions = [r if set("NSEWUL") & set(r) else str(int(float(r))).zfill(2) for r in regions]
print(regions)