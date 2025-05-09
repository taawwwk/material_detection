import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# true, pred 불러오기
y_true = np.load('/Users/kimtaewook/Documents/GitHub/material_detection/results/y_test_flat_1745739007.npy')
y_pred = np.load('/Users/kimtaewook/Documents/GitHub/material_detection/results/y_test_cnn_1745744508.npy')

# 클래스 이름
class_names = ['steel','wooden','glass']

# 혼동행렬 계산 및 시각화
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)

disp.plot()
plt.title('MLP Confusion Matrix')
plt.show()