import glob



files = glob.glob('benchmark/*.log')

# 2021-12-06 17:50:10,766 - WARNING - images: ../data/benchmark_ellipses
# 2021-12-06 17:50:10,766 - WARNING - n_proj: 32
# 2021-12-06 17:50:10,766 - WARNING - noise_pow: 40.0
def parse(f):
    f_dict = {

    }
    for line in open(f).readlines():
        if 'CRITICAL' in line:
            info = line.replace('\n','').split('-')[4]
            print(info)     

for f in files:
    #print('\n', f)
    parse(f)

print()