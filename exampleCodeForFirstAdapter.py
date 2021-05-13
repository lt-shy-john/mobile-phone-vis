    
import numpy as np
import matplotlib.pyplot as plt

import firstAdapter as adap
companyNames, companyScores = adap.getCompanyNameAndScore()
matrix, companyNames = adap.getEarlyAdapMatrix()

print(np.sum(matrix, axis=0))

plt.bar(companyNames, companyScores)
plt.show()