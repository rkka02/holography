import slmpy
import numpy as np
# import hologram # 필요시 사용
from PIL import Image
import time
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from lucam import Lucam
# --- 전역 변수 또는 설정 ---
ROI_RECT = (400, 200, 1500, 1800) # y_start, x_start, y_end, x_end
ROI_HEIGHT = ROI_RECT[2] - ROI_RECT[0]
ROI_WIDTH = ROI_RECT[3] - ROI_RECT[1]

SLM_RES_X, SLM_RES_Y = 1920, 1080 # 예시 값, 실제 SLM에 맞게 수정 (slm.getSize()로 덮어쓰여짐)

# "가벼운 뉴럴 네트워크" (SLM 패턴 -> 예측 ROI)
class SystemModel(nn.Module):
    def __init__(self, slm_height, slm_width, roi_height, roi_width):
        super(SystemModel, self).__init__()
        # 입력: [N, 1, slm_height, slm_width] (0-1 정규화된 SLM 패턴)
        # 출력: [N, 1, roi_height, roi_width] (0-1 정규화된 예측 ROI)

        # 간단한 CNN 예시 (채널 수, 커널 크기, 패딩 등은 조절 가능)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2) # SLM 해상도 유지
        self.relu1 = nn.ReLU()
        # 해상도를 점진적으로 줄이거나, 바로 ROI 크기로 맞출 수 있음
        # 예: AdaptiveAvgPool2d 사용
        self.pool1 = nn.AdaptiveAvgPool2d((roi_height // 2, roi_width // 2)) # 중간 크기로 풀링

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        
        # 최종 ROI 크기로 변환하는 레이어
        # 예: Upsample 후 Conv 또는 ConvTranspose2d
        self.upsample = nn.Upsample(size=(roi_height, roi_width), mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1) # 최종 채널 1
        self.sigmoid = nn.Sigmoid() # 출력을 0-1 범위로

    def forward(self, slm_pattern_normalized):
        # slm_pattern_normalized: [N, 1, slm_height, slm_width], 0-1 범위
        x = self.relu1(self.conv1(slm_pattern_normalized))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.upsample(x)
        x = self.conv3(x)
        predicted_roi = self.sigmoid(x) # [N, 1, roi_height, roi_width], 0-1 범위
        return predicted_roi

# PyTorch 모델 정의 (0-255 사이의 값을 생성)
class SLMGenerator(nn.Module):
    def __init__(self, height, width):
        super(SLMGenerator, self).__init__()
        # 내부적으로 학습되는 파라미터 (범위 제약 없음, 초기값을 0 근처로)
        self.raw_pattern = nn.Parameter(torch.randn(height, width) * 0.1) # 초기값을 작게 하여 sigmoid 출력이 0.5 근처가 되도록

    def forward(self):
        # Sigmoid를 통해 0~1 범위로 만들고, 255를 곱해 0~255 범위로 스케일링
        return torch.sigmoid(self.raw_pattern) * 255.0 # 255.0으로 하여 float 연산 유지

def preprocess_image_to_tensor(image_path_or_array, target_size=None, is_captured=False):
    if isinstance(image_path_or_array, str):
        img = Image.open(image_path_or_array).convert('L')
    elif isinstance(image_path_or_array, np.ndarray):
        if image_path_or_array.ndim == 3:
             img = Image.fromarray(image_path_or_array).convert('L')
        elif image_path_or_array.dtype == np.uint16: # 16비트 이미지 처리 추가 (Lucam 경우)
            img_8bit = (image_path_or_array / 256).astype(np.uint8) # 16bit -> 8bit 단순 스케일링
            img = Image.fromarray(img_8bit).convert('L')
        else:
             img = Image.fromarray(image_path_or_array)
    else:
        raise ValueError("Input must be a file path string or a numpy array.")

    if is_captured:
        img = img.crop((ROI_RECT[1], ROI_RECT[0], ROI_RECT[3], ROI_RECT[2]))
    
    if target_size:
        img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = transform(img)
    return tensor.squeeze(0)

def pattern_tensor_to_slm_screen(pattern_tensor):
    """
    PyTorch 패턴 텐서(0-255 범위 실수)를 SLM에 표시할 수 있는
    8비트 정수형 numpy 배열로 변환.
    """
    # 텐서 값을 반올림하고, CPU로 옮기고, numpy 배열로 변환 후 uint8로 타입 캐스팅
    # pattern_tensor는 SLMGenerator의 forward()에서 이미 0-255 범위로 조정됨
    slm_screen_np = pattern_tensor.round().clone().detach().cpu().numpy().astype(np.uint8)
    return slm_screen_np

# --- 주 실행 로직 ---
if __name__ == '__main__':
    NUM_ITERATIONS = 100
    LEARNING_RATE = 0.1 # 학습률은 실험을 통해 조절 (sigmoid 출력 범위 고려하여 조정 가능)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # SLM_LUT() 함수는 최적화에 직접 사용되지 않음
    # phase_val_lut, slm_digit_lut = SLM_LUT() # 이 줄은 필요 없음

    slm = None
    camera = Lucam()

    try:
        print("Initializing SLM...")
        slm = slmpy.SLMdisplay(monitor=0)
        SLM_RES_X, SLM_RES_Y = slm.getSize()
        print(f"SLM Resolution: {SLM_RES_X}x{SLM_RES_Y}")

        print("Initializing Camera...")
        # camera = Lucam() # 실제 카메라 사용 시 주석 해제
        # print(f"Camera initialized: {camera.ShowPreview()}")
        # time.sleep(1)

        target_image_tensor = preprocess_image_to_tensor(
            'cat.jpg',
            target_size=(ROI_HEIGHT, ROI_WIDTH)
        ).to(DEVICE)
        print(f"Target image tensor shape: {target_image_tensor.shape}")

        slm_generator = SLMGenerator(SLM_RES_Y, SLM_RES_X).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(slm_generator.parameters(), lr=LEARNING_RATE)

        for iteration in range(NUM_ITERATIONS):
            optimizer.zero_grad()

            current_pattern_tensor = slm_generator() # 0-255 범위의 실수형 텐서

            slm_screen_to_display = pattern_tensor_to_slm_screen(current_pattern_tensor)
            
            slm.updateArray(slm_screen_to_display)
            time.sleep(1) # SLM 반응 및 카메라 노출 시간 (조절 필요)

            captured_image_np = camera.TakeSnapshot()
            captured_roi_tensor = preprocess_image_to_tensor(
                captured_image_np,
                is_captured=True
            ).to(DEVICE)
            
            loss = criterion(captured_roi_tensor, target_image_tensor)
            print(loss)
            print(captured_roi_tensor.requires_grad)
            loss.backward()
            optimizer.step()

            print(f"Iteration {iteration+1}/{NUM_ITERATIONS}, Loss: {loss.item():.6f}")

            if (iteration + 1) % 20 == 0:
                # 현재 SLM에 표시된 화면(정수값) 저장
                Image.fromarray(slm_screen_to_display).save(f"test/slm_display_iter_{iteration+1}.png")
                # 현재 캡처된 ROI 저장
                # captured_roi_tensor는 0-1 범위이므로 255 곱해서 저장
                Image.fromarray(
                    (captured_roi_tensor.cpu().numpy() * 255).astype(np.uint8)
                ).save(f"test/captured_roi_iter_{iteration+1}.png")

        optimized_pattern_tensor = slm_generator().detach().cpu()
        print("Optimization finished.")
        # final_slm_screen = pattern_tensor_to_slm_screen(optimized_pattern_tensor)
        # Image.fromarray(final_slm_screen).save("optimized_slm_screen_0_255.png")
        # np.save("optimized_pattern_0_255.npy", optimized_pattern_tensor.numpy()) # 0-255 실수값 패턴

    except Exception as e:
        print(f"An error occurred: {e}")
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")
    finally:
        if slm:
            print("Closing SLM...")
            slm.close()
        # if camera: # ...
            # print("Closing Camera...")
            # camera.Release()