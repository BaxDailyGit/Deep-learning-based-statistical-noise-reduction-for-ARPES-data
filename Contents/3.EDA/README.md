# 3. EDA & 전처리
###### 다음은 ARPES 실험 데이터를 가져와 분석하는 코드입니다.

## 모듈 가져오기
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
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
<p align="center"><img src="https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-ARPES-data/assets/99312529/031064f8-1378-443e-909c-b0d40782c231" width="40%" height="40%"></p>

## 상수 정의
```python
h = 6.626e-34 # Planck constant (m^2 kg/s)
hbar = h/(2*(np.pi)) # Dirac's constant
m = 9.109e-31 # electron mass (kg)
hv = 29 # 빛의 에너지  (eV)
wf = 4.43 # 일함수 (eV)
```
## matrix 행(x축), 열(y축) 정보
```python
delta_ke = 0.001 # kinetic Energy(eV)의 delta값
start_ke = 23.885 #kinetic Energy(eV)의 시작값
ke_unit = 'eV'
kinetic_energy = np.linspace(start_ke, start_ke + delta_ke * matrix.shape[0], matrix.shape[0])

delta_theta = 0.041096 # 각도(Θ)의 delta값
start_theta = -16.3795 # 각도(Θ)의 시작값
theta_unit = 'slit deg'
theta = np.linspace(start_theta, start_theta + delta_theta * matrix.shape[1], matrix.shape[1])
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

<p align="center"><img src="https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-ARPES-data/assets/99312529/dbadcf61-2538-4f19-8e85-4f2b568e5826" width="40%" height="40%"></p>

## Binding Energy(eV) 계산




#### $$E_k = hν − φ − E_B 이므로$$

## $$E_k − hν − φ = − E_B $$

###### • $hν$ : 빛 에너지
###### • $φ$ : 샘플 표면 작용함수(surface work function) 즉, 일함수
###### • $E_k$ : 광전자 운동 에너지
###### • $E_B$ : 전자가 방출되기 전에 가지고 있던 결합 에너지
###### +) $E_k − hν − φ$ 를 $E − E_F$ 로 표현하기도 한다.
```python
binding_energy = hv - wf - kinetic_energy
binding_energy = (-1)*binding_energy
# − E_B 즉, E − E_F를 축으로 사용한다.
```
## K 계산




##### $$ħk_{||} = \sqrt {2mE_k}sin{θ}  이므로$$

## $$k_{||} = \frac{\sqrt {2mE_k}}{ħ}sin{θ}$$
###### • $ħK_{||}$ : 표면 평면에 대한 운동량
###### • $ħ$ :  Dirac's constant 
###### • $K_{||}$ : 파동수 m^{-1}
###### • $m$ : 전자 질량 (kg)
###### • $E_k (J)$ = $E_k (eV)$ * 1.602176634e-19 : 광전자 운동 에너지 (J)   
```python
#단위 변환 (eV -> J)
kinetic_energy_J = kinetic_energy * 1.602176634e-19
```

##### 다만 K가 kinetic_energy와 theta 두 변수에 영향을 받기 때문에 2차원 배열입니다. 
##### 최종적으로 학습시킬 데이터는 binding_energy,K에 대한 intensity를 나타낸 2차원 데이터이기 때문에 2차원인 K를 그래프의 축으로 할 수가 없습니다. 
##### 즉, 1차원의 K를 새롭게 만들어야 합니다.

<p align="center"><img src="https://user-images.githubusercontent.com/99312529/237040878-1d805813-8d8f-44b0-b8ae-99ac6d42d6ea.png" width="40%" height="40%"></p>


##### 이때 K와 theta는 동일하게 대응하면 안됩니다.
###### 만약 theta를 기준으로 하나의 theta에 대응하는 K와 각 kinetic_energy에 해당하는 intesnsity를 구하는 식으로 그래프를 만들면 theta가 sin함수 안에 있기 때문에 K의 간격이 점점 좁아져 0°에서 멀어질수록 그래프는 찌그러지게 될것입니다.

<p align="center"><img src="https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-ARPES-data/assets/99312529/6e7ca785-7a70-4987-a116-c1637cac6559">


##### 1 ) 양쪽을 kinetic_energy의 최댓값과 theta 양끝값을 대입해 구하고 그 사이를 균등한 간격으로 linspace합니다.
##### 2 ) 새롭게 만든 각 K의 해당하는 kinetic_energy와 theta는 기존 데이터에 없기 때문에 주변값들을 활용하여 보간해야합니다.
 
```python
theta = np.linspace(start_theta, start_theta + delta_theta * matrix.shape[1], matrix.shape[1])
K_first = np.sqrt(2*m*max(kinetic_energy_J)) * np.sin(np.deg2rad(start_theta)) / hbar
K_last = np.sqrt(2*m*max(kinetic_energy_J)) * np.sin(np.deg2rad(max(theta))) / hbar
K = np.linspace(K_first, K_last, theta.size)
K=K*(10**(-10)) # 단위 m -> Å
 
# 2차원 보간 함수 생성
interp_func = interp2d(K, binding_energy, matrix, kind='linear') # kind = 'linear': 선형 보간, 'cubic': 3차 스플라인 보간, 'quintic': 5차 스플라인 보간

# 2차원 보간 함수를 이용하여 보간된 새로운 행렬 생성
interp_EB_K_matrix = interp_func(K, binding_energy)
```
##### 코드를 보면 k 양쪽끝을 구하고 theta개수만큼 linspace하는데 theta개수만큼 만드는 이유는 개수를 늘리면 보간해야하는 데이터가 많아져 동시에 정확하지 않은 데이터가 많아질 확률이 올라가고, 개수를 줄이면 결국 가로축의 개수가 적어진다는 의미이므로 해상도가 낮아집니다.



## binding_energy와 K 그래프 그리기
```python
fig, ax = plt.subplots()
im = ax.imshow(interp_EB_K_matrix , aspect='auto',cmap='jet',origin='lower',extent=[K[0],K[-1] , binding_energy[0], binding_energy[-1]])
ax.set_xlabel('K (Å$^{-1}$)')
ax.set_ylabel(' $E-E_F$ ({0})'.format(ke_unit))
cbar =fig.colorbar(im)
cbar.set_label('intensity')
```

<p align="center"><img src="https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-ARPES-data/assets/99312529/e243e79b-bf2f-4d5c-9e43-5ccba5d71788" width="40%" height="40%"></p>

## CSV 파일로 저장
```python
# kinetic_energy와 theta 그래프 CSV 파일로 저장
df = pd.DataFrame(matrix, columns=theta, index=kinetic_energy)
df.columns.name = 'theta ({0})'
df.index.name = 'Kinetic Energy ({0})'.format(ke_unit)
df.to_csv('Ek_theta_matrix.csv')
 
# binding_energy와 K 그래프 CSV 파일로 저장
df = pd.DataFrame(interp_EB_K_matrix , columns=K, index=binding_energy)
df.columns.name = 'K (Å$^{-1}$)'
df.index.name = '$E-E_F$ ({0})'.format(ke_unit)
df.to_csv('interp_Eb_K_matrix.csv')
```
## 데이터셋 구성
##### 위와 같은 방식으로 처리된 TaSe2_GK, TaSe2_MK, WSe2 입니다. 
##### 설명 추가 예정

```python
# CSV 파일 경로 및 파일명 리스트
csv_files = ['/content/drive/MyDrive/ARPES/TaSe2_GK.csv', '/content/drive/MyDrive/ARPES/TaSe2_MK.csv', '/content/drive/MyDrive/ARPES/WSe2.csv']

# 시작값과 간격값 설정
start_be = [-0.28271, -0.316606, -2.09679]  # 파일별 시작값
delta_be = [0.0005, 0.0005, 0.00159001]  # 파일별 간격값
start_K = [-0.755169, -0.449906, -0.578732]  # 파일별 시작값
delta_K = [0.00138108, 0.00140804, 0.00166317]  # 파일별 간격값

class DataProcessor:
    def __init__(self, csv_files, start_be, delta_be, start_K, delta_K):
        self.csv_files = csv_files
        self.start_be = start_be
        self.delta_be = delta_be
        self.start_K = start_K
        self.delta_K = delta_K
        self.matrix_list = []
        self.new_matrix_list = []

    def read_csv_files(self):
        for file, start_be_val, delta_be_val, start_K_val, delta_K_val in zip(
            self.csv_files, self.start_be, self.delta_be, self.start_K, self.delta_K
        ):
            data = np.genfromtxt(file, delimiter=',')
            matrix = np.transpose(data)
            self.matrix_list.append(matrix)

    def make_new_matrix_list(self):
        num_plots = len(self.matrix_list)
        fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(20, 5))

        for i in range(num_plots):
            be_unit = 'eV'
            binding_energy = np.linspace(
                self.start_be[i], self.start_be[i] + self.delta_be[i] * self.matrix_list[i].shape[0],
                self.matrix_list[i].shape[0]
            )
            K_unit = '(Å$^{-1}$)'
            K = np.linspace(
                self.start_K[i], self.start_K[i] + self.delta_K[i] * self.matrix_list[i].shape[1],
                self.matrix_list[i].shape[1]
            )

            interp_func = interp2d(K, binding_energy, self.matrix_list[i], kind='linear')
            new_K = np.linspace(K.min(), K.max(), 600)
            new_binding_energy = np.linspace(binding_energy.min(), binding_energy.max(), 600)
            new_matrix = interp_func(new_K, new_binding_energy)

            self.new_matrix_list.append(new_matrix)

            im = axes[i].imshow(
                self.new_matrix_list[i], extent=[new_K[0], new_K[-1], new_binding_energy[0], new_binding_energy[-1]],
                aspect='auto', cmap='jet', origin='lower'
            )
            axes[i].set_title(self.csv_files[i])
            axes[i].set_xlabel('K (Å$^{-1}$)')
            axes[i].set_ylabel('$E-E_F$ ({0})'.format(be_unit))
            cbar = fig.colorbar(im, ax=axes[i])
            cbar.set_label('Intensity')

        plt.tight_layout()
        plt.show()
        
        
        
processor = DataProcessor(csv_files, start_be, delta_be, start_K, delta_K)
processor.read_csv_files()
processor.make_new_matrix_list()
```
<p align="center"><img src="https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-ARPES-data/assets/99312529/fd978b0f-ea9e-4fe7-9278-c41066432d01" width="100%" height="100%"></p>

## data augmentation
##### TaSe2_GK 우선적으로 노이즈를 입혀보았습니다.
##### 설명 추가 예정

```python

# 원본 이미지를 일단 new_matrix_list[0]으로 저장
original_image = new_matrix_list[0]


# 노이즈가 있는 이미지 생성 (원본 이미지에 가우시안 노이즈 추가)
mean = 0
stddev = 0.01 # 노이즈의 표준 편차 조절
noisy_image = []


# 가우시안 노이즈를 사용한 데이터 증강
num_augmented_images = 4
augmented_noisy_image_list = []
for i in range(num_augmented_images):
    np.random.seed(i)  # 시드 값을 반복문 변수 i로 설정
    augmented_noisy_image = original_image + np.random.normal(mean, stddev, original_image.shape)
    augmented_noisy_image_list.append(augmented_noisy_image)

def plot_new_matrix_list(matrix_list,new_K):
    num_plots = len(augmented_noisy_image_list)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(30, 5))
    
    for i in range(num_plots):
        #print(matrix_list[0].shape[1])
        # 데이터 변경 (시작값과 간격값 적용)
        be_unit = 'eV'
        binding_energy = np.linspace(start_be[0], start_be[0] + delta_be[0] * matrix_list[0].shape[0], matrix_list[i].shape[0])
    
        im = axes[i].imshow(matrix_list[i], extent=[new_K[0], new_K[-1], binding_energy[0], binding_energy[-1]], aspect='auto', cmap='gray', origin='lower')
        axes[i].set_xlabel('K (Å$^{-1}$)')
        axes[i].set_ylabel('$E-E_F$ ({0})'.format(be_unit))
        cbar = fig.colorbar(im, ax=axes[i])
        cbar.set_label('Intensity')
    
    plt.tight_layout()
    plt.show()
    
plot_new_matrix_list(augmented_noisy_image_list,new_K_list[0]) 
```
<p align="center"><img src="https://github.com/BaxDailyGit/Deep-learning-based-statistical-noise-reduction-for-ARPES-data/assets/99312529/1866dbda-50ff-40ee-8487-7e040d89892c" width="100%" height="100%"></p>
