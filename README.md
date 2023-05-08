Deep learning based statistical noise reduction for ARPES data
=========

# 목차

### [ 1. 들어가며](https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-multidimensional-spectral-data/tree/main/Contents/1.%EB%93%A4%EC%96%B4%EA%B0%80%EB%A9%B0)   
### [ 2. ARPES data 이해하기](https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-multidimensional-spectral-data/tree/main/Contents/2.ARPES%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0)   
### [ 3. EDA & 전처리](https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-multidimensional-spectral-data/tree/main/Contents/3.EDA)   
### [ 4. Training](https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-multidimensional-spectral-data/tree/main/Contents/4.ML)   
### [ 5. 참고](https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-multidimensional-spectral-data/tree/main/Contents/%EC%B0%B8%EA%B3%A0)   
 

---------

# 1. 들어가며

###### 물리학특수연구를 진행한다. (23.05 ~ 23.06.)   
###### 연구 목표 : ARPES data를 이용하여 물질의 특성을 알 수가 있는데 280k의 동일한 온도에서 수집한 ARPES 데이터와 학습 데이터셋의 무작위 생성을 통해 과적합 없이 신경망을 학습시킵니다. 이 신경망은 노이즈가 있는 데이터에 대해 고유 정보를 보존하면서 데이터의 노이즈를 제거하여 보다 양질의 데이터를 분석할 수가 있습니다.   
###### 현재까지의 결과 : 우선 ARPES에 대한 기초지식을 공부하고 데이터를 분석 및 처리에 적합한 형태로 만들기 위해 전처리 과정중에 있습니다. 또한 학습데이터를 어떻게 확보할지 만약 데이터에 노이즈를 주는 식이면 어떻게 줄지 (random generation method으로 과적합 방지), 최선의 딥러닝 학습방법과 모델 선정에 앞서 관련 지식을 공부하고 있습니다.   
###### 남은 기간 동안의 계획 : 전반적인 파이프라인을 오류 없이 빠르게 구축한 후 보다 좋은 모델을 선택, 테스트데이터 추가, Data Augmentation , 다양한 방법을 통한 데이터셋을 확장하는 등으로 나머지 시간을 할애 할 것 같습니다.    
---------


# ARPES data 이해하기

##### ARPES는 광전효과를 기반으로 합니다. 


#### 광전효과

###### 광전효과 : 금속 등의 물질이 한계 진동수(문턱 진동수)보다 큰 진동수를 가진 (따라서 높은 에너지를 가진) 전자기파를 흡수했을 때 전자를 내보내는 현상이다. 이 때 방출되는 전자를 **광전자**라 하는데, 보통 전자와 성질이 다르지는 않지만 빛에 의해 방출되는 전자이기 때문에 붙여진 이름이다.
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/a/a6/Photoelectric_effect_in_a_solid_-_diagram.svg/413px-Photoelectric_effect_in_a_solid_-_diagram.svg.png" width="20%" height="20%"></p>   

###### [이미지 출처: https://ko.wikipedia.org/wiki/광전효과]


## ARPES

#### ARPES : 물질의 밴드구조를 직접적으로 관측하는 장비
(angle resolved photoemission spectroscopy - 각도 분해 광전자 분광법)
<p align="center"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/ARPES_analyzer_cross_section.svg/300px-ARPES_analyzer_cross_section.svg.png" width="40%" height="40%"></p>

###### [이미지 출처:https://fr.wikipedia.org/wiki/Spectroscopie_photo%C3%A9lectronique_r%C3%A9solue_en_angle]
#### -간단설명-
#### 1) ARPES를 사용하여 감지된 각 전자의 운동 에너지 $E_k$과 방출 각도 $(θ, φ)$를 기록합니다. 
#### 2) $E_k$와 $(θ, φ)$를 가지고 운동학적 특성을 이용하여 전자가 물질로부터 방출되기 전에 가지고 있던 결합 에너지 $E_B$와 결정운동량 $ħk$를 추정할 수 있습니다.
#### 3) 바로 $E_B$와  $ħk$를 가지고 조합하여 만든 밴드 구조를 통해 고체의 성질(정보)을 알 수 있습니다.<br>
<br>
<br>

---------------------------------------
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
###### • $ħk_{||}$ : 표면 평면에 대한 운동량

###### 다만 K가 kinetic_energy와 theta 두 변수에 영향을 받기 때문에 2차원 배열입니다. 그래서 K를 축으로 사용하기 위해서는 1차원이어야 합니다.

<p align="center"><img src="https://user-images.githubusercontent.com/99312529/236875192-fd6d69d2-aff4-4436-82b8-89694ce8d8ac.png" width="100%" height="100%"></p>

###### 양쪽을 kinetic_energy의 최댓값과 theta 양끝값을 대입해 구하고 그 사이를 균등한 간격으로 하고 scipy.interpolate(보간법)을 이용해 자료 사이의 빠진 부분인 kinetic_energy, theta, binding_energy, intensity를 유추하면 될 것 같습니다. 다차원이다보니 헷갈린데 조금더 생각해보고 코드를 수정하여 올리겠습니다.

```python
'''
theta = np.linspace(start_theta, start_theta + delta_theta * matrix.shape[1], matrix.shape[1]) # matrix.shape[0]은 열 개수를 의미 #kinetic_energy와 동일하게 생성
K = np.zeros((matrix.shape[0], matrix.shape[1])) # 2차원 배열 K를 만들고, 이 배열의 크기는 matrix의 행, 열 개수 동일, 모든 요소가 0
for i in range(matrix.shape[0]):
    K[i, :] = ((2 * m * kinetic_energy[i])** 0.5 / h) * np.sin(np.radians(theta))
    # K[i,:] 즉, i번째 행의 모든 열에 대해 값을 할당하는 반복문.
'''
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
df = pd.DataFrame(matrix, columns=theta, index=kinetic_energy)
df.index.name = 'Knetic Energy ({0})'.format(ke_unit)
df.columns.name = 'theta ({0})'.format(theta_unit)
df.to_csv('matrix.csv')

df = pd.DataFrame(matrix, columns=K, index=binding_energy_energy) # K가 2차원이라 안됌.
df.index.name = 'Binding Energy ({0})'.format(ke_unit)
df.columns.name = 'K ({0})'.format(theta_unit)
df.to_csv('matrix.csv')
```

----------
# 참고   

###### 1.Angle-resolved photoemission studies of quantum materials
###### 양자 물질의 각도 분해 광전자 방출 연구(ARPES)
###### https://ui.adsabs.harvard.edu/abs/2021RvMP...93b5006S/abstract

###### 2.Deep learning-based statistical noise reduction for multidimensional spectral data
###### 다차원 스펙트럼 데이터에 대한 딥러닝 기반 통계적 노이즈 감소
###### https://ui.adsabs.harvard.edu/abs/2021RScI...92g3901K/abstract

