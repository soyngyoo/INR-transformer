import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb  # 옵션: 학습 모니터링용

def train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, scaler, device, epoch):
    model.train()
    total_loss = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch + 1}") as pbar:
        for batch_idx, (coords, labels, pixel_values) in enumerate(pbar):
            # 데이터를 device로 이동
            coords = coords.to(device)
            labels = labels.to(device)
            pixel_values = pixel_values.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad(set_to_none=True)  # 메모리 효율적
            
            # Mixed Precision Training
            with autocast():
                pred_pixel_values = model(coords, labels)
                loss = loss_fn(pred_pixel_values.squeeze(), pixel_values)
            
            # 역전파 및 가중치 업데이트 (mixed precision)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Learning rate 업데이트 (만약 OneCycleLR 등을 사용한다면)
            if scheduler is not None:
                scheduler.step()
            
            # 손실값 기록
            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Progress bar 업데이트
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{current_lr:.6f}'
            })
            
            # Wandb 로깅 (옵션)
            if wandb.run is not None:
                wandb.log({
                    'batch_loss': loss.item(),
                    'learning_rate': current_lr,
                })
    
    return total_loss / len(train_loader)

def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for coords, labels, pixel_values in val_loader:
            coords = coords.to(device)
            labels = labels.to(device)
            pixel_values = pixel_values.to(device)
            
            pred_pixel_values = model(coords, labels)
            loss = loss_fn(pred_pixel_values.squeeze(), pixel_values)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)
