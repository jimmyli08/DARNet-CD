import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import os
import warnings
import numpy as np
import time

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser, metadata = get_parser_with_args()
opt = parser.parse_args()
opt.batch_size = 8

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

path = 'model_weight.pt'  


print(path)
model = torch.load(path)
model = model.module
c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()

with torch.no_grad():
    tbar = tqdm(test_loader,ncols=80)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)
        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        labels_np = labels.data.cpu().numpy()
        preds_np = cd_preds.data.cpu().numpy()
        tp = np.sum((labels_np == 1) & (preds_np == 1))
        tn = np.sum((labels_np == 0) & (preds_np == 0))
        fp = np.sum((labels_np == 0) & (preds_np == 1))
        fn = np.sum((labels_np == 1) & (preds_np == 0))

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp
        

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)
IOU = tp/(tp+fp+fn)
OA = (tp+tn)/(tp+fp+tn+fn)
p0 = OA
pe = ((tp+fp)*(tp+fn)+(fp+tn)*(fn+tn))/(tp+fp+tn+fn)**2
Kappa = (p0-pe)/(1-pe)

print('Precision: {}\nRecall: {}\nF1-Score: {} \nIOU:{}'.format(P, R, F1,IOU))
print('OA: {}\nKappa: {}'.format(OA,Kappa))
