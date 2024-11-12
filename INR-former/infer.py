import torch
import matplotlib.pyplot as plt

def generate_image(model, label, image_size=28, device='cuda'):
    model.eval()  # 평가 모드로 설정
    
    with torch.no_grad():  # 그래디언트 계산 비활성화
        # 1. 그리드 좌표 생성
        x = torch.linspace(-1, 1, image_size)
        y = torch.linspace(-1, 1, image_size)
        grid_y, grid_x = torch.meshgrid(y, x)
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)
        
        # 2. 라벨 준비
        labels = torch.full((coords.shape[0],), label, dtype=torch.long).to(device)
        
        # 3. 배치 처리를 위한 설정
        batch_size = 1024  # GPU 메모리에 따라 조정
        pixel_values = []
        
        # 4. 배치 단위로 처리
        for i in range(0, coords.shape[0], batch_size):
            batch_coords = coords[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            
            # 픽셀값 예측
            batch_pixels = model(batch_coords, batch_labels)
            pixel_values.append(batch_pixels)
        
        # 5. 결과 합치기
        pixel_values = torch.cat(pixel_values, dim=0)
        
        # 6. 이미지 형태로 변환
        image = pixel_values.reshape(image_size, image_size).cpu()
        
        return image

# 여러 숫자에 대한 이미지 생성 및 시각화
def visualize_generated_images(model, device='cuda', num_samples=10):
    plt.figure(figsize=(20, 4))
    
    for i in range(num_samples):
        plt.subplot(2, 5, i+1)
        image = generate_image(model, label=i, device=device)
        plt.imshow(image, cmap='gray')
        plt.title(f'Label: {i}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 보간 (interpolation) 기능 추가
def interpolate_images(model, label1, label2, steps=10, image_size=28, device='cuda'):
    model.eval()
    images = []
    
    with torch.no_grad():
        # 1. 그리드 좌표 준비
        x = torch.linspace(-1, 1, image_size)
        y = torch.linspace(-1, 1, image_size)
        grid_y, grid_x = torch.meshgrid(y, x)
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1).to(device)
        
        # 2. 라벨 interpolation
        alphas = torch.linspace(0, 1, steps)
        
        for alpha in alphas:
            # 라벨 임베딩 interpolation
            label_tensor1 = torch.tensor([label1], device=device)
            label_tensor2 = torch.tensor([label2], device=device)
            
            # 각 좌표에 대해 동일한 interpolated 라벨 적용
            labels = torch.full((coords.shape[0],), label1, dtype=torch.long).to(device)
            
            # 배치 처리
            pixel_values = []
            batch_size = 1024
            
            for i in range(0, coords.shape[0], batch_size):
                batch_coords = coords[i:i+batch_size]
                batch_labels = labels[i:i+batch_size]
                batch_pixels = model(batch_coords, batch_labels)
                pixel_values.append(batch_pixels)
            
            pixel_values = torch.cat(pixel_values, dim=0)
            image = pixel_values.reshape(image_size, image_size).cpu()
            images.append(image)
    
    # 시각화
    plt.figure(figsize=(20, 4))
    for i, img in enumerate(images):
        plt.subplot(2, steps//2, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'α: {alphas[i]:.2f}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return images
   