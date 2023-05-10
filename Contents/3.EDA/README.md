# EDA & 전처리
###### 다음은 ARPES 실험 데이터를 가져와 분석하는 코드입니다.

## 모듈 가져오기
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```
## CSV 파일 읽고 전치
```python
data = np.genfromtxt('Cut280K_rNx.csv', delimiter=',')
matrix = np.transpose(data)
```
## 로우데이터 시각화
```python
plt.imshow(matrix,origin='lower')
plt.xlabel('theta (cell)')
plt.ylabel('Kinetic Energy (cell)')
plt.colorbar() #옆에 컬러바
```
<p align="center"><img src="https://user-images.githubusercontent.com/99312529/236664917-2a4e96b5-ca3b-482c-b3a8-ff9d37f7429a.png" width="40%" height="40%"></p>


## 상수 정의
```python
h = 6.626e-34 # Planck constant (m^2 kg/s)
m = 9.109e-31 # electron mass (kg)
hv = 29 # 빛의 에너지  (eV)
wf = 4.43 # 일함수 (eV)
```
## matrix 행(x축), 열(y축) 정보
```python
delta_ke = 0.001 # kinetic Energy(eV)의 delta값
start_ke = 23.885 #kinetic Energy(eV)의 시작값
ke_unit = 'eV'
delta_theta = 0.0410959 # 각도(Θ)의 delta값
start_theta = -17.9795 # 각도(Θ)의 시작값
theta_unit = 'slit deg'
```
## Binding Energy(eV) 계산



#### $$E_k = hν − φ − E_B 이므로$$

## $$E_B = hν − φ − E_k$$

###### • $hν$ : 빛 에너지
###### • $φ$ : 샘플 표면 작용함수(surface work function) 즉, 일함수
###### • $E_k$ : 광전자 운동 에너지
###### • $E_B$ : 전자가 방출되기 전에 가지고 있던 결합 에너지
```python
kinetic_energy = np.linspace(start_ke, start_ke + delta_ke * matrix.shape[0], matrix.shape[0]) #matrix.shape[0]은 행 개수를 의미
#linspace 함수는 start_ke에서 시작해 start_ke+delta_ke*matrix.shape[0]값까지 동일한 간격으로 matrix.shape[0]개의 값을 생성 배열에 저장
binding_energy = hv - wf - kinetic_energy
```
## K 계산




##### $$ħk_{||} = \sqrt {2mE_k}sin{θ}  이므로$$

## $$k_{||} = \frac{\sqrt {2mE_k}}{ħ}sin{θ}$$
###### • $ħK_{||}$ : 표면 평면에 대한 운동량
###### • $ħ$ : 플랑크 상수 (m^2 kg/s)
###### • $K_{||}$ : 파동수 m^{-1}
###### • $m$ : 전자 질량 (kg)
###### • $E_k (J)$ = $E_k (eV)$ * 1.602176634e-19 : 광전자 운동 에너지 (J)   
```python
#단위 변환
kinetic_energy_J = kinetic_energy * 1.602176634e-19
```

##### 다만 K가 kinetic_energy와 theta 두 변수에 영향을 받기 때문에 2차원 배열입니다. 
##### 최종적으로 학습시킬 데이터는 binding_energy,K에 대한 intensity를 나타낸 3차원 데이터이기 때문에 2차원인 K를 그래프의 축으로 할 수가 없습니다. 
##### 즉, 1차원의 K를 새롭게 만들어야 합니다. (1차원의 K와 그 K에 해당하는 kinetic_energy,theta의 정보가 담긴 K_inf도 만들어야 합니다.)

<p align="center"><img src="https://user-images.githubusercontent.com/99312529/237040878-1d805813-8d8f-44b0-b8ae-99ac6d42d6ea.png" width="40%" height="40%"></p>


##### 이때 K와 theta는 동일하게 대응하면 안됩니다.
###### 만약 theta를 기준으로 하나의 theta에서 각 kinetic_energy에 해당하는 intesnsity를 구하는 식으로 그래프를 만들면 theta가 sin함수 안에 있기 때문에 K의 간격이 점점 좁아져 0°에서 멀어질수록 그래프는 찌그러지게 될것입니다.

<p align="center"><img src="https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-ARPES-data/assets/99312529/6e7ca785-7a70-4987-a116-c1637cac6559">


##### 1 ) 양쪽을 kinetic_energy의 최댓값과 theta 양끝값을 대입해 구하고 그 사이를 균등한 간격으로 linspace합니다.
##### 2 ) 새롭게 만든 각 K의 해당하는 kinetic_energy와 theta는 기존 데이터에 없기 때문에 주변값들을 활용하여 보간해야합니다.
###### scipy.interpolate(보간법)을 kinetic_energy, theta, intensity를 유추하면 됩니다. 


```python
K_first = np.sqrt(2*m*max(kinetic_energy_J)) * np.sin(np.deg2rad(start_theta)) / h
K_last = np.sqrt(2*m*max(kinetic_energy_J)) * np.sin(np.deg2rad(max(theta))) / h
K = np.linspace(K_first, K_last, matrix.shape[1])
'''
보간하는 코드 추가하기 
'''
```
##### 코드를 보면 k 양쪽끝을 구하고 theta개수만큼 linspace하는데 theta개수만큼 만드는 이유는 개수를 늘리면 보간해야하는 데이터가 많아져 동시에 정확하지 않은 데이터가 많아질 확률이 올라가고, 개수를 줄이면 결국 가로축의 개수가 적어진다는 의미이므로 해상도가 낮아집니다.
###### 다차원이다보니 헷갈린데 코드를 수정해보고 올리겠습니다.


## kinetic_energy와 theta 그래프 그리기
```python
fig, ax = plt.subplots()
im = ax.imshow(matrix, extent=[theta.min(), theta.max(), kinetic_energy.min(), kinetic_energy.max()], aspect='auto', cmap='jet',origin='lower',interpolation='nearest')
ax.set_xlabel('theta ({0})'.format(theta_unit))
ax.set_ylabel('kinetic Energy ({0})'.format(ke_unit))
cbar = fig.colorbar(im)
cbar.set_label('intensity')
```

<p align="center"><img src="https://user-images.githubusercontent.com/99312529/236664946-89b49e3c-c386-4d0a-8081-4337a1f270df.png" width="40%" height="40%"></p>

## binding_energy와 K 그래프 그리기
```python
'''
'''
```



## CSV 파일로 저장
```python
df = pd.DataFrame(matrix, columns=theta, index=kinetic_energy)
df.index.name = 'Knetic Energy ({0})'.format(ke_unit)
df.columns.name = 'theta ({0})'.format(theta_unit)
df.to_csv('matrix.csv')

df = pd.DataFrame(matrix, columns=K, index=binding_energy_energy) 
df.index.name = 'Binding Energy ({0})'.format(ke_unit)
df.columns.name = 'K (1/m)'
df.to_csv('matrix.csv')
```
