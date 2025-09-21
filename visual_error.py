import numpy as np
import os
# from osgeo import gdal
from skimage import io

def binary_accuracy(pred, label):
    w, h = pred.shape
    result = np.zeros((w, h, 3))
    pred = (pred >= 0.5)
    label = (label >= 0.5)

    TP = pred * label
    FP = pred * (1 - label)
    FN = (1 - pred) * label
    TN = (1 - pred) * (1 - label)

    # TP
    result[:, :, 0] = np.where(TP == 1, 255, result[:, :, 0])
    result[:, :, 1] = np.where(TP == 1, 255, result[:, :, 1])
    result[:, :, 2] = np.where(TP == 1, 255, result[:, :, 2])

    # FP
    result[:, :, 0] = np.where(FP == 1, 255, result[:, :, 0])
    result[:, :, 1] = np.where(FP == 1, 0, result[:, :, 1])
    result[:, :, 2] = np.where(FP == 1, 0, result[:, :, 2])

    # FN
    result[:, :, 0] = np.where(FN == 1, 0, result[:, :, 0])
    result[:, :, 1] = np.where(FN == 1, 0, result[:, :, 1])
    result[:, :, 2] = np.where(FN == 1, 255, result[:, :, 2])

    # TN
    result[:, :, 0] = np.where(TN == 1, 0, result[:, :, 0])
    result[:, :, 1] = np.where(TN == 1, 0, result[:, :, 1])
    result[:, :, 2] = np.where(TN == 1, 0, result[:, :, 2])

    return result


if __name__ == '__main__':
    # import glob
    # import tqdm

    Path = r'E:\results_HRSCD_results\BCD_truth'                  #GT
    predList = r'E:\change'             ###predict results
    save = r'E:\change_error'

    names = []
    accs = []
    ious = []
    f1s = []
    precisions = []
    recalls = []
    for finename in os.listdir(predList):
        predictPath = predList + '/' + finename
        gtPath = Path + '/' + finename[:-4] + '.png'
        outName = save+ '/' + finename[:-4] + '.png'
        pred = io.imread(predictPath)
        gt = io.imread(gtPath)
        gt = np.where(gt > 0, 1, 0).astype(np.uint8)
        pred = np.where(pred > 0, 1, 0).astype(np.uint8)
        result = binary_accuracy(pred, gt)

        io.imsave(outName, result.astype(np.uint8))
