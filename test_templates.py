from hiddini import ObservationsTemplateCosSim
import numpy as np
import scipy.linalg

type_templates = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                           [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                           [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]])
num_dimensions = type_templates.shape[1]
chord_templates = np.dstack([scipy.linalg.circulant(i) for i in type_templates]).reshape(num_dimensions, -1).T

observer = ObservationsTemplateCosSim(chord_templates)
obs = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]]).T
probs = observer(obs)
print(probs)