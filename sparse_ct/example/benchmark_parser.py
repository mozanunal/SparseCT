import glob



files = glob.glob('benchmark/*.log')


def parse(f):
    f_dict = {

    }
    for line in open(f).readlines():
        line = line.split('-')[4].replace('\n','')
        if 'Avg' in line:
            _, _, mse, psnr, ssim = line.split(' ')
            mse = mse.replace('MSE:', '')
            psnr = psnr.replace('PSNR:', '')
            ssim = ssim.replace('SSIM:', '')
            print(mse, psnr, ssim, end=' ' )
        if 'Std' in line:
            pass
            # _, _, mse, psnr, ssim = line.split(' ')
            # print(mse, psnr, ssim)

            

for f in files:
    print('\n', f, end=' ')
    parse(f)

print()