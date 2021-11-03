import os
import string
import subprocess
import concurrent.futures
import tqdm
from termcolor import colored

scenes = [ 'scene0000_00', 'scene0051_00', 'scene0089_01', 'scene0126_02', 'scene0181_01', 'scene0223_00', 'scene0267_00', 'scene0320_01', 'scene0381_00', 'scene0523_00',  'scene0568_00',
'scene0000_01', 'scene0051_01', 'scene0089_02', 'scene0127_00', 'scene0181_02', 'scene0223_01', 'scene0268_00', 'scene0320_02', 'scene0381_01', 'scene0523_01',  'scene0568_01',
'scene0000_02', 'scene0051_02', 'scene0090_00', 'scene0127_01', 'scene0181_03', 'scene0223_02', 'scene0268_01', 'scene0320_03', 'scene0381_02', 'scene0523_02',  'scene0568_02',
'scene0001_00', 'scene0051_03', 'scene0091_00', 'scene0128_00', 'scene0182_00', 'scene0224_00', 'scene0268_02', 'scene0321_00', 'scene0382_00', 'scene0524_00',  'scene0569_00',
'scene0001_01', 'scene0052_00', 'scene0092_00', 'scene0129_00', 'scene0182_01', 'scene0225_00', 'scene0269_00', 'scene0322_00', 'scene0382_01', 'scene0524_01',  'scene0569_01',
'scene0002_00', 'scene0052_01', 'scene0092_01', 'scene0130_00', 'scene0182_02', 'scene0226_00', 'scene0269_01', 'scene0323_00', 'scene0383_00', 'scene0525_00',  'scene0570_00',
'scene0002_01', 'scene0052_02', 'scene0092_02', 'scene0131_00', 'scene0183_00', 'scene0226_01', 'scene0269_02', 'scene0323_01', 'scene0383_01', 'scene0525_01',  'scene0570_01',
'scene0003_00', 'scene0053_00', 'scene0092_03', 'scene0131_01', 'scene0184_00', 'scene0227_00', 'scene0280_00', 'scene0324_00', 'scene0383_02', 'scene0525_02',  'scene0570_02',
'scene0003_01', 'scene0054_00', 'scene0092_04', 'scene0131_02', 'scene0185_00', 'scene0228_00', 'scene0280_01', 'scene0324_01', 'scene0384_00', 'scene0526_00',  'scene0571_00',
'scene0003_02', 'scene0055_00', 'scene0093_00', 'scene0132_00', 'scene0186_00', 'scene0229_00', 'scene0280_02', 'scene0325_00', 'scene0385_00', 'scene0526_01',  'scene0571_01',
'scene0004_00', 'scene0055_01', 'scene0093_01', 'scene0132_01', 'scene0186_01', 'scene0229_01', 'scene0281_00', 'scene0325_01', 'scene0385_01', 'scene0527_00',  'scene0572_00',
'scene0005_00', 'scene0055_02', 'scene0093_02', 'scene0132_02', 'scene0187_00', 'scene0229_02', 'scene0282_00', 'scene0326_00', 'scene0385_02', 'scene0528_00',  'scene0572_01',
'scene0005_01', 'scene0056_00', 'scene0094_00', 'scene0133_00', 'scene0187_01', 'scene0230_00', 'scene0282_01', 'scene0327_00', 'scene0386_00', 'scene0528_01',  'scene0572_02',
'scene0006_00', 'scene0056_01', 'scene0095_00', 'scene0134_00', 'scene0188_00', 'scene0231_00', 'scene0282_02', 'scene0328_00', 'scene0387_00', 'scene0529_00',  'scene0573_00',
'scene0006_01', 'scene0057_00', 'scene0095_01', 'scene0134_01', 'scene0189_00', 'scene0231_01', 'scene0283_00', 'scene0329_00', 'scene0387_01', 'scene0529_01',  'scene0573_01',
'scene0006_02', 'scene0057_01', 'scene0096_00', 'scene0134_02', 'scene0190_00', 'scene0231_02', 'scene0284_00', 'scene0329_01', 'scene0387_02', 'scene0529_02',  'scene0574_00',
'scene0007_00', 'scene0058_00', 'scene0096_01', 'scene0135_00', 'scene0191_00', 'scene0232_00', 'scene0285_00', 'scene0329_02', 'scene0388_00', 'scene0530_00',  'scene0574_01',
'scene0008_00', 'scene0058_01', 'scene0096_02', 'scene0136_00', 'scene0191_01', 'scene0232_01', 'scene0286_00', 'scene0340_00', 'scene0388_01', 'scene0531_00',  'scene0574_02',
'scene0009_00', 'scene0059_00', 'scene0097_00', 'scene0136_01', 'scene0191_02', 'scene0232_02', 'scene0286_01', 'scene0340_01', 'scene0389_00', 'scene0532_00',  'scene0575_00',
'scene0009_01', 'scene0059_01', 'scene0098_00', 'scene0136_02', 'scene0192_00', 'scene0233_00', 'scene0286_02', 'scene0340_02', 'scene0390_00', 'scene0532_01',  'scene0575_01',
'scene0009_02', 'scene0059_02', 'scene0098_01', 'scene0137_00', 'scene0192_01', 'scene0233_01', 'scene0286_03', 'scene0341_00', 'scene0391_00', 'scene0533_00',  'scene0575_02',
'scene0010_00', 'scene0060_00', 'scene0099_00', 'scene0137_01', 'scene0192_02', 'scene0234_00', 'scene0287_00', 'scene0341_01', 'scene0392_00', 'scene0533_01',  'scene0576_00',
'scene0010_01', 'scene0060_01', 'scene0099_01', 'scene0137_02', 'scene0193_00', 'scene0235_00', 'scene0288_00', 'scene0342_00', 'scene0392_01', 'scene0534_00',  'scene0576_01',
'scene0011_00', 'scene0061_00', 'scene0100_00', 'scene0138_00', 'scene0193_01', 'scene0236_00', 'scene0288_01', 'scene0343_00', 'scene0392_02', 'scene0534_01',  'scene0576_02',
'scene0011_01', 'scene0061_01', 'scene0100_01', 'scene0139_00', 'scene0194_00', 'scene0236_01', 'scene0288_02', 'scene0344_00', 'scene0393_00', 'scene0535_00',  'scene0577_00',
'scene0012_00', 'scene0062_00', 'scene0100_02', 'scene0140_00', 'scene0195_00', 'scene0237_00', 'scene0289_00', 'scene0344_01', 'scene0393_01', 'scene0536_00',  'scene0578_00',
'scene0012_01', 'scene0062_01', 'scene0101_00', 'scene0140_01', 'scene0195_01', 'scene0237_01', 'scene0289_01', 'scene0345_00', 'scene0393_02', 'scene0536_01',  'scene0578_01',
'scene0012_02', 'scene0062_02', 'scene0101_01', 'scene0141_00', 'scene0195_02', 'scene0238_00', 'scene0290_00', 'scene0345_01', 'scene0394_00', 'scene0536_02',  'scene0578_02',
'scene0013_00', 'scene0063_00', 'scene0101_02', 'scene0141_01', 'scene0196_00', 'scene0238_01', 'scene0291_00', 'scene0346_00', 'scene0394_01', 'scene0537_00',  'scene0579_00',
'scene0013_01', 'scene0064_00', 'scene0101_03', 'scene0141_02', 'scene0197_00', 'scene0239_00', 'scene0291_01', 'scene0346_01', 'scene0395_00', 'scene0538_00',  'scene0579_01',
'scene0013_02', 'scene0064_01', 'scene0101_04', 'scene0142_00', 'scene0197_01', 'scene0239_01', 'scene0291_02', 'scene0347_00', 'scene0395_01', 'scene0539_00',  'scene0579_02',
'scene0014_00', 'scene0065_00', 'scene0101_05', 'scene0142_01', 'scene0197_02', 'scene0239_02', 'scene0292_00', 'scene0347_01', 'scene0395_02', 'scene0539_01',  'scene0580_00',
'scene0015_00', 'scene0065_01', 'scene0102_00', 'scene0143_00', 'scene0198_00', 'scene0240_00', 'scene0292_01', 'scene0347_02', 'scene0396_00', 'scene0539_02',  'scene0580_01',
'scene0016_00', 'scene0065_02', 'scene0102_01', 'scene0143_01', 'scene0199_00', 'scene0241_00', 'scene0293_00', 'scene0348_00', 'scene0396_01', 'scene0540_00',  'scene0581_00',
'scene0016_01', 'scene0066_00', 'scene0103_00', 'scene0143_02', 'scene0200_00', 'scene0241_01', 'scene0293_01', 'scene0348_01', 'scene0396_02', 'scene0540_01',  'scene0581_01',
'scene0016_02', 'scene0067_00', 'scene0103_01', 'scene0144_00', 'scene0200_01', 'scene0241_02', 'scene0294_00', 'scene0348_02', 'scene0397_00', 'scene0540_02',  'scene0581_02',
'scene0017_00', 'scene0067_01', 'scene0104_00', 'scene0144_01', 'scene0200_02', 'scene0242_00', 'scene0294_01', 'scene0349_00', 'scene0397_01', 'scene0541_00',  'scene0582_00',
'scene0017_01', 'scene0067_02', 'scene0105_00', 'scene0145_00', 'scene0201_00', 'scene0242_01', 'scene0294_02', 'scene0349_01', 'scene0398_00', 'scene0541_01',  'scene0582_01',
'scene0017_02', 'scene0068_00', 'scene0105_01', 'scene0146_00', 'scene0201_01', 'scene0242_02', 'scene0295_00', 'scene0350_00', 'scene0398_01', 'scene0541_02',  'scene0582_02',
'scene0018_00', 'scene0068_01', 'scene0105_02', 'scene0146_01', 'scene0201_02', 'scene0243_00', 'scene0295_01', 'scene0350_01', 'scene0399_00', 'scene0542_00',  'scene0583_00',
'scene0019_00', 'scene0069_00', 'scene0106_00', 'scene0146_02', 'scene0202_00', 'scene0244_00', 'scene0296_00', 'scene0350_02', 'scene0399_01', 'scene0543_00',  'scene0583_01',
'scene0019_01', 'scene0070_00', 'scene0106_01', 'scene0147_00', 'scene0203_00', 'scene0244_01', 'scene0296_01', 'scene0351_00', 'scene0500_00', 'scene0543_01',  'scene0583_02',
'scene0020_00', 'scene0071_00', 'scene0106_02', 'scene0147_01', 'scene0203_01', 'scene0245_00', 'scene0297_00', 'scene0351_01', 'scene0500_01', 'scene0543_02',  'scene0584_00',
'scene0020_01', 'scene0072_00', 'scene0107_00', 'scene0148_00', 'scene0203_02', 'scene0246_00', 'scene0297_01', 'scene0352_00', 'scene0501_00', 'scene0544_00',  'scene0584_01',
'scene0021_00', 'scene0072_01', 'scene0108_00', 'scene0149_00', 'scene0204_00', 'scene0247_00', 'scene0297_02', 'scene0352_01', 'scene0501_01', 'scene0545_00',  'scene0584_02',
'scene0022_00', 'scene0072_02', 'scene0109_00', 'scene0150_00', 'scene0204_01', 'scene0247_01', 'scene0298_00', 'scene0352_02', 'scene0501_02', 'scene0545_01',  'scene0585_00',
'scene0022_01', 'scene0073_00', 'scene0109_01', 'scene0150_01', 'scene0204_02', 'scene0248_00', 'scene0299_00', 'scene0353_00', 'scene0502_00', 'scene0545_02',  'scene0585_01',
'scene0023_00', 'scene0073_01', 'scene0110_00', 'scene0150_02', 'scene0205_00', 'scene0248_01', 'scene0299_01', 'scene0353_01', 'scene0502_01', 'scene0546_00',  'scene0586_00',
'scene0024_00', 'scene0073_02', 'scene0110_01', 'scene0151_00', 'scene0205_01', 'scene0248_02', 'scene0300_00', 'scene0353_02', 'scene0502_02', 'scene0547_00',  'scene0586_01',
'scene0024_01', 'scene0073_03', 'scene0110_02', 'scene0151_01', 'scene0205_02', 'scene0249_00', 'scene0300_01', 'scene0354_00', 'scene0503_00', 'scene0547_01',  'scene0586_02',
'scene0024_02', 'scene0074_00', 'scene0111_00', 'scene0152_00', 'scene0206_00', 'scene0250_00', 'scene0301_00', 'scene0355_00', 'scene0504_00', 'scene0547_02',  'scene0587_00',
'scene0025_00', 'scene0074_01', 'scene0111_01', 'scene0152_01', 'scene0206_01', 'scene0250_01', 'scene0301_01', 'scene0355_01', 'scene0505_00', 'scene0548_00',  'scene0587_01',
'scene0025_01', 'scene0074_02', 'scene0111_02', 'scene0152_02', 'scene0206_02', 'scene0250_02', 'scene0301_02', 'scene0356_00', 'scene0505_01', 'scene0548_01',  'scene0587_02',
'scene0025_02', 'scene0075_00', 'scene0112_00', 'scene0153_00', 'scene0207_00', 'scene0251_00', 'scene0302_00', 'scene0356_01', 'scene0505_02', 'scene0548_02',  'scene0587_03',
'scene0026_00', 'scene0076_00', 'scene0112_01', 'scene0153_01', 'scene0207_01', 'scene0252_00', 'scene0302_01', 'scene0356_02', 'scene0505_03', 'scene0549_00',  'scene0588_00',
'scene0027_00', 'scene0077_00', 'scene0112_02', 'scene0154_00', 'scene0207_02', 'scene0253_00', 'scene0303_00', 'scene0357_00', 'scene0505_04', 'scene0549_01',  'scene0588_01',
'scene0027_01', 'scene0077_01', 'scene0113_00', 'scene0155_00', 'scene0208_00', 'scene0254_00', 'scene0303_01', 'scene0357_01', 'scene0506_00', 'scene0550_00',  'scene0588_02',
'scene0027_02', 'scene0078_00', 'scene0113_01', 'scene0155_01', 'scene0209_00', 'scene0254_01', 'scene0303_02', 'scene0358_00', 'scene0507_00', 'scene0551_00',  'scene0588_03',
'scene0028_00', 'scene0078_01', 'scene0114_00', 'scene0155_02', 'scene0209_01', 'scene0255_00', 'scene0304_00', 'scene0358_01', 'scene0508_00', 'scene0552_00',  'scene0589_00',
'scene0029_00', 'scene0078_02', 'scene0114_01', 'scene0156_00', 'scene0209_02', 'scene0255_01', 'scene0305_00', 'scene0358_02', 'scene0508_01', 'scene0552_01',  'scene0589_01',
'scene0029_01', 'scene0079_00', 'scene0114_02', 'scene0157_00', 'scene0210_00', 'scene0255_02', 'scene0305_01', 'scene0359_00', 'scene0508_02', 'scene0553_00',  'scene0589_02',
'scene0029_02', 'scene0079_01', 'scene0115_00', 'scene0157_01', 'scene0210_01', 'scene0256_00', 'scene0306_00', 'scene0359_01', 'scene0509_00', 'scene0553_01',  'scene0590_00',
'scene0030_00', 'scene0080_00', 'scene0115_01', 'scene0158_00', 'scene0211_00', 'scene0256_01', 'scene0306_01', 'scene0370_00', 'scene0509_01', 'scene0553_02',  'scene0590_01',
'scene0030_01', 'scene0080_01', 'scene0115_02', 'scene0158_01', 'scene0211_01', 'scene0256_02', 'scene0307_00', 'scene0370_01', 'scene0509_02', 'scene0554_00',  'scene0591_00',
'scene0030_02', 'scene0080_02', 'scene0116_00', 'scene0158_02', 'scene0211_02', 'scene0257_00', 'scene0307_01', 'scene0370_02', 'scene0510_00', 'scene0554_01',  'scene0591_01',
'scene0031_00', 'scene0081_00', 'scene0116_01', 'scene0159_00', 'scene0211_03', 'scene0258_00', 'scene0307_02', 'scene0371_00', 'scene0510_01', 'scene0555_00',  'scene0591_02',
'scene0031_01', 'scene0081_01', 'scene0116_02', 'scene0170_00', 'scene0212_00', 'scene0259_00', 'scene0308_00', 'scene0371_01', 'scene0510_02', 'scene0556_00',  'scene0592_00',
'scene0031_02', 'scene0081_02', 'scene0117_00', 'scene0170_01', 'scene0212_01', 'scene0259_01', 'scene0309_00', 'scene0372_00', 'scene0511_00', 'scene0556_01',  'scene0592_01',
'scene0032_00', 'scene0082_00', 'scene0118_00', 'scene0170_02', 'scene0212_02', 'scene0260_00', 'scene0309_01', 'scene0373_00', 'scene0511_01', 'scene0557_00',  'scene0593_00',
'scene0032_01', 'scene0083_00', 'scene0118_01', 'scene0171_00', 'scene0213_00', 'scene0260_01', 'scene0310_00', 'scene0373_01', 'scene0512_00', 'scene0557_01',  'scene0593_01',
'scene0033_00', 'scene0083_01', 'scene0118_02', 'scene0171_01', 'scene0214_00', 'scene0260_02', 'scene0310_01', 'scene0374_00', 'scene0513_00', 'scene0557_02',  'scene0594_00',
'scene0034_00', 'scene0084_00', 'scene0119_00', 'scene0172_00', 'scene0214_01', 'scene0261_00', 'scene0310_02', 'scene0375_00', 'scene0514_00', 'scene0558_00',  'scene0595_00',
'scene0034_01', 'scene0084_01', 'scene0120_00', 'scene0172_01', 'scene0214_02', 'scene0261_01', 'scene0311_00', 'scene0375_01', 'scene0514_01', 'scene0558_01',  'scene0596_00',
'scene0034_02', 'scene0084_02', 'scene0120_01', 'scene0173_00', 'scene0215_00', 'scene0261_02', 'scene0312_00', 'scene0375_02', 'scene0515_00', 'scene0558_02',  'scene0596_01',
'scene0035_00', 'scene0085_00', 'scene0121_00', 'scene0173_01', 'scene0215_01', 'scene0261_03', 'scene0312_01', 'scene0376_00', 'scene0515_01', 'scene0559_00',  'scene0596_02',
'scene0035_01', 'scene0085_01', 'scene0121_01', 'scene0173_02', 'scene0216_00', 'scene0262_00', 'scene0312_02', 'scene0376_01', 'scene0515_02', 'scene0559_01',  'scene0597_00',
'scene0036_00', 'scene0086_00', 'scene0121_02', 'scene0174_00', 'scene0217_00', 'scene0262_01', 'scene0313_00', 'scene0376_02', 'scene0516_00', 'scene0559_02',  'scene0597_01',
'scene0036_01', 'scene0086_01', 'scene0122_00', 'scene0174_01', 'scene0218_00', 'scene0263_00', 'scene0313_01', 'scene0377_00', 'scene0516_01', 'scene0560_00',  'scene0597_02',
'scene0037_00', 'scene0086_02', 'scene0122_01', 'scene0175_00', 'scene0218_01', 'scene0263_01', 'scene0313_02', 'scene0377_01', 'scene0517_00', 'scene0561_00',  'scene0598_00',
'scene0038_00', 'scene0087_00', 'scene0123_00', 'scene0176_00', 'scene0219_00', 'scene0264_00', 'scene0314_00', 'scene0377_02', 'scene0517_01', 'scene0561_01',  'scene0598_01',
'scene0038_01', 'scene0087_01', 'scene0123_01', 'scene0177_00', 'scene0220_00', 'scene0264_01', 'scene0315_00', 'scene0378_00', 'scene0517_02', 'scene0562_00',  'scene0598_02',
'scene0038_02', 'scene0087_02', 'scene0123_02', 'scene0177_01', 'scene0220_01', 'scene0264_02', 'scene0316_00', 'scene0378_01', 'scene0518_00', 'scene0563_00',  'scene0599_00',
'scene0039_00', 'scene0088_00', 'scene0124_00', 'scene0177_02', 'scene0220_02', 'scene0265_00', 'scene0317_00', 'scene0378_02', 'scene0519_00', 'scene0564_00',  'scene0599_01',
'scene0039_01', 'scene0088_01', 'scene0124_01', 'scene0178_00', 'scene0221_00', 'scene0265_01', 'scene0317_01', 'scene0379_00', 'scene0520_00', 'scene0565_00',  'scene0599_02',
'scene0050_00', 'scene0088_02', 'scene0125_00', 'scene0179_00', 'scene0221_01', 'scene0265_02', 'scene0318_00', 'scene0380_00', 'scene0520_01', 'scene0566_00',  'scene0522_00',
'scene0050_01', 'scene0088_03', 'scene0126_00', 'scene0180_00', 'scene0222_00', 'scene0266_00', 'scene0319_00', 'scene0380_01', 'scene0521_00', 'scene0567_00',  'scene0567_01',
'scene0050_02', 'scene0089_00', 'scene0126_01', 'scene0181_00', 'scene0222_01', 'scene0266_01', 'scene0320_00', 'scene0380_02' ]



def run_bash_cmd(scene):
    cmd = 'python scannet-reader.py --filename scannet_render_rgb/scans/{}/{}.sens --output_path scannet-rgb/scans/{}/ --export_color_images'.format(scene, scene, scene)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result

with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    results = list(tqdm.tqdm(executor.map(run_bash_cmd, scenes), total=len(scenes)))
    print(colored('\n  Successfully ran {} cmds'.format(results.count(True)), 'green'))
    print(colored('  Error in {} cmds'.format(results.count(False)), 'red'))
