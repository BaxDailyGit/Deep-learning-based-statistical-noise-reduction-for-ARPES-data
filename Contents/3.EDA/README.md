# EDA
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
h = 6.626e-34 # Planck constant
m = 9.109e-31 # electron mass
hv = 29 # 빛의 에너지 (eV)
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
###### • $ħk_{||}$ : 표면 평면에 대한 결정운동량
###### • 결정 구조의 이산 평면 주기성 때문에, 광전자방출 과정 전체에서 $k_{||}$는 보존됩니다(평면 상호 격자 벡터 $G_{||}$를 기준으로).
###### • 수직 구성 성분 $k_⊥$는 표면을 통과하는 동안 보존되지 않지만, 일부 가정하에서 추정할 수 있습니다
```python
theta = np.linspace(start_theta, start_theta + delta_theta * matrix.shape[1], matrix.shape[1]) # matrix.shape[0]은 열 개수를 의미 #kinetic_energy와 동일하게 생성
K = np.zeros((matrix.shape[0], matrix.shape[1])) # 2차원 배열 K를 만들고, 이 배열의 크기는 matrix의 행, 열 개수 동일, 모든 요소가 0
for i in range(matrix.shape[0]):
    K[i, :] = ((2 * m * kinetic_energy[i])** 0.5 / h) * np.sin(np.radians(theta))
    # K[i,:] 즉, i번째 행의 모든 열에 대해 값을 할당하는 반복문.
```
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
fig, ax = plt.subplots()
im = ax.imshow(matrix, extent=[K.min(), K.max(), binding_energy.min(), binding_energy.max()], aspect='auto', cmap='jet',origin='lower',interpolation='nearest')
ax.set_xlabel('K (1/m)')
ax.set_ylabel('Binding Energy ({0})'.format(ke_unit))
cbar = fig.colorbar(im)
cbar.set_label('intensity')
plt.show()
```
<p align="center"><img src="https://user-images.githubusercontent.com/99312529/236664977-e6dba0d6-d8bb-412b-978e-8c45a3a32af1.png" width="40%" height="40%"></p>



## CSV 파일로 저장
```python
df = pd.DataFrame(matrix, columns=theta, index=binding_energy)
df.index.name = 'Binding Energy ({0})'.format(ke_unit)
df.columns.name = 'theta ({0})'.format(theta_unit)
df.to_csv('matrix.csv')
```
