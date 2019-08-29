import os
from drawMM import drawMM
import numpy as np

def drawConfusionMatrix(output_folder, TPon = False, TNon = False, FPon = False, FNon = False, dupon = False, dupN = 1):
    if not os.path.isdir(output_folder):
        raise Exception("%s does not exist!" %(output_folder))
    # load x_wanted.csv
    with open("%s/x_wanted.csv" %(output_folder), "r") as f:
        x_wanted = f.read().split('\n')
    # TP
    if TPon:
        TProot = output_folder + '/' + 'TP'
        if not os.path.isdir(TProot):
            os.mkdir(TProot)
    with open("%s/%s_TP.csv" %(output_folder, output_folder), "r") as f:
        TPs = f.read().split('\n')
    TPsum = np.zeros(csvstr2array(x_wanted[0]).shape)
    TPnum = 0
    # avoid repeat
    generated = set()
    for TP in TPs:
        if TP != '':
            row = int(TP)
            if x_wanted[row] not in generated:
                if TPon:
                    drawer = drawMM(csvstr2array(x_wanted[row]), imageName = "%s/x_wanted_%d.png" %(TProot, row))
                    generated.add(x_wanted[row])
                    if dupon:
                        drawer.dupNbyM(dupN, dupN)
                    drawer.save()
                TPsum += csvstr2array(x_wanted[row])
                TPnum += 1
    if TPon:
        print("Finish generating True Positive images in %s" %(TProot))
    # TN
    if TNon:
        TNroot = output_folder + '/' + 'TN'
        if not os.path.isdir(TNroot):
            os.mkdir(TNroot)
    with open("%s/%s_TN.csv" %(output_folder, output_folder), "r") as f:
        TNs = f.read().split('\n')
    TNsum = np.zeros(csvstr2array(x_wanted[0]).shape)
    TNnum = 0
    # avoid repeat
    generated = set()
    for TN in TNs:
        if TN != '':
            row = int(TN)
            if x_wanted[row] not in generated:
                if TNon:
                    drawer = drawMM(csvstr2array(x_wanted[row]), imageName = "%s/x_wanted_%d.png" %(TNroot, row))
                    generated.add(x_wanted[row])
                    if dupon:
                        drawer.dupNbyM(dupN, dupN)
                    drawer.save()
                TNsum += csvstr2array(x_wanted[row])
                TNnum += 1
    if TNon:
        print("Finish generating True Negative images in %s" %(TNroot))
    # FP
    if FPon:
        FProot = output_folder + '/' + 'FP'
        if not os.path.isdir(FProot):
            os.mkdir(FProot)
    with open("%s/%s_FP.csv" %(output_folder, output_folder), "r") as f:
        FPs = f.read().split('\n')
    FPsum = np.zeros(csvstr2array(x_wanted[0]).shape)
    FPnum = 0
    # avoid repeat
    generated = set()
    for FP in FPs:
        if FP != '':
            row = int(FP)
            if x_wanted[row] not in generated:
                if FPon:
                    drawer = drawMM(csvstr2array(x_wanted[row]), imageName = "%s/x_wanted_%d.png" %(FProot, row))
                    generated.add(x_wanted[row])
                    if dupon:
                        drawer.dupNbyM(dupN, dupN)
                    drawer.save()
                FPsum += csvstr2array(x_wanted[row])
                FPnum += 1
    if FPon:
        print("Finish generating False Positive images in %s" %(FProot))
    # FN
    if FNon:
        FNroot = output_folder + '/' + 'FN'
        if not os.path.isdir(FNroot):
            os.mkdir(FNroot)
    with open("%s/%s_FN.csv" %(output_folder, output_folder), "r") as f:
        FNs = f.read().split('\n')
    FNsum = np.zeros(csvstr2array(x_wanted[0]).shape)
    FNnum = 0
    # avoid repeat
    generated = set()
    for FN in FNs:
        if FN != '':
            row = int(FN)
            if x_wanted[row] not in generated:
                if FNon:
                    drawer = drawMM(csvstr2array(x_wanted[row]), imageName = "%s/x_wanted_%d.png" %(FNroot, row))
                    generated.add(x_wanted[row])
                    if dupon:
                        drawer.dupNbyM(dupN, dupN)
                    drawer.save()
                FNsum += csvstr2array(x_wanted[row])
                FNnum += 1
    if FNon:
        print("Finish generating False Negative images in %s" %(FNroot))
    # generate Confusion Matrix
    print("Generating Confusion Matrix......")
    generateConfusionMatrix(output_folder, TPsum, TPnum, TNsum, TNnum, FPsum, FPnum, FNsum, FNnum)

def csvstr2array(csvstr):
    return np.array(list(map(float, csvstr.split(','))))

def generateConfusionMatrix(output_folder, TPsum, TPnum, TNsum, TNnum, FPsum, FPnum, FNsum, FNnum):
    ncell = len(TPsum)
    with open("%s/confusion_matrix.csv" %(output_folder), "w", newline = "") as f:
        # header
        for c in range(ncell):
            f.write(",") # skip the first column
            f.write("cell_%d" %(c + 1)) # cell 1 to 15
        f.write('\n')
        # TP
        f.write("True Positive")
        for c in range(ncell):
            f.write(",") # skip the first column
            f.write(str(TPsum[c])) # cell 1 to 15
        f.write('\n')
        # FP
        f.write("False Positive")
        for c in range(ncell):
            f.write(",") # skip the first column
            f.write(str(FPsum[c])) # cell 1 to 15
        f.write('\n')
        # TN
        f.write("True Negative")
        for c in range(ncell):
            f.write(",") # skip the first column
            f.write(str(TNsum[c])) # cell 1 to 15
        f.write('\n')
        # FN
        f.write("False Negative")
        for c in range(ncell):
            f.write(",") # skip the first column
            f.write(str(FNsum[c])) # cell 1 to 15
        f.write('\n')
        ## Percentage
        f.write("Percentage\n")
        # TP
        f.write("True Positive")
        for c in range(ncell):
            f.write(",") # skip the first column
            f.write("%.2f%%" %(TPsum[c]/TPnum*100)) # cell 1 to 15
        f.write('\n')
        # FP
        f.write("False Positive")
        for c in range(ncell):
            f.write(",") # skip the first column
            f.write("%.2f%%" %(FPsum[c]/FPnum*100)) # cell 1 to 15
        f.write('\n')
        # TN
        f.write("True Negative")
        for c in range(ncell):
            f.write(",") # skip the first column
            f.write("%.2f%%" %(TNsum[c]/TNnum*100)) # cell 1 to 15
        f.write('\n')
        # FN
        f.write("False Negative")
        for c in range(ncell):
            f.write(",") # skip the first column
            f.write("%.2f%%" %(FNsum[c]/FNnum*100)) # cell 1 to 15
        f.write('\n')
        # Confusion Matrix
        f.write('\n')
        f.write('\n')
        f.write(', Positive, Negative')
        f.write('\n')
        f.write('True, %.2f%%, %.2f%%' %(TPnum/(TPnum+FNnum)*100, TNnum/(TNnum+FPnum)*100))
        f.write('\n')
        f.write('False, %.2f%%, %.2f%%' %(FPnum/(TNnum+FPnum)*100, FNnum/(TPnum+FNnum)*100))

    print("Finish generating Confusion Matrix in %s/confusion_matrix.csv" %(output_folder))

if __name__ == '__main__':
    output_folder = input("Please input the path to the output_folder, e.g. PSV_bg_2: ").strip()
    # True Positive on/off
    TPonraw = input("Want to plot True Positive? [Y/N] ")
    if TPonraw.lower() == 'y':
        TPon = True
    else:
        TPon = False
    # True Negative on/off
    TNonraw = input("Want to plot True Negative? [Y/N] ")
    if TNonraw.lower() == 'y':
        TNon = True
    else:
        TNon = False
    # False Positive on/off
    FPonraw = input("Want to plot False Positive? [Y/N] ")
    if FPonraw.lower() == 'y':
        FPon = True
    else:
        FPon = False
    # False Negative on/off
    FNonraw = input("Want to plot False Negative? [Y/N] ")
    if FNonraw.lower() == 'y':
        FNon = True
    else:
        FNon = False
    # Need duplication?
    dup = input("Want to duplicate the image by n times? [Y/N] ")
    if dup.lower() == 'y':
        dupon = True
        dupN = int(input("n equals: ").strip())
    else:
        dupon = False
        dupN = 1
    drawConfusionMatrix(output_folder, TPon, TNon, FPon, FNon, dupon, dupN)