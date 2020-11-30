import glob



files = glob.glob('benchmark/*.log')


def parse(f):
    f_dict = {

    }
    for line in open(f).readlines():
        line = line.split('-')[4].replace('\n','')
        if 'Avg' in line:
            _, _, _, psnr, ssim = line.split(' ')
            psnr = psnr.replace('PSNR:', '')
            ssim = ssim.replace('SSIM:', '')
        if 'Std' in line:
            _, _, _, psnr_std, ssim_std = line.split(' ')
            psnr_std = psnr_std.replace('PSNR:', '')
            ssim_std = ssim_std.replace('SSIM:', '')
            print("{:.2f}+{:.2f} {:.2f}+{:.2f}".format(
                    float(psnr), float(psnr_std),
                    float(ssim)*100, float(ssim_std)*100,
                ), 
                end=' ' )

            

for f in files:
    print('\n', f, end=' ')
    parse(f)

print()